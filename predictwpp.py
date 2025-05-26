import sqlite3
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss, classification_report
import numpy as np
from typing import Optional, Tuple, List
import os
from tqdm import tqdm
import random

# --- 乱数シードの固定（結果の再現性を確保するため） ---
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # GPU使用時の決定性を確保するための追加設定（パフォーマンスに影響する可能性あり）
    # tf.config.experimental.enable_op_determinism()

# シードを固定して実行

# --- Keras/TensorFlow のインポート ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

# 出力幅を広げる設定
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)

"""## 2: データ読み込み・前処理関数の定義 (load_and_process_pbp)"""

def load_and_process_pbp(db_path: str, limit_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    SQLiteデータベースからプレイバイプレイデータを読み込み、データ型を処理し、
    ホームチームの勝敗情報を計算して元のデータに結合します。
    (チャンク読み込みとプログレスバー表示に対応)
    """
    print(f"--- Starting data loading and processing ---")
    print(f"Database path: {db_path}")
    if limit_rows:
        print(f"Row limit: {limit_rows}")
    else:
        print("Row limit: None (loading all rows)")

    try:
        print("Connecting to database...")
        conn = sqlite3.connect(db_path)
        print("Database connection successful.")

        total_rows_to_load = limit_rows
        if total_rows_to_load is None:
            print("Determining total row count for progress bar...")
            count_query = "SELECT COUNT(*) FROM play_by_play;"
            total_rows_to_load = pd.read_sql_query(count_query, conn).iloc[0, 0]
            print(f"Total rows in table: {total_rows_to_load}")

        query = """
        SELECT
            game_id, eventnum, eventmsgtype, eventmsgactiontype,
            period, pctimestring,
            homedescription, neutraldescription, visitordescription,
            score, scoremargin
        FROM
            play_by_play
        """
        if limit_rows:
            query += f" LIMIT {limit_rows};"
        else:
            query += ";"

        chunk_size = 100000
        print(f"Loading data in chunks of {chunk_size} rows...")
        iterator = pd.read_sql_query(query, conn, chunksize=chunk_size)
        num_chunks = (total_rows_to_load + chunk_size - 1) // chunk_size

        list_of_dfs = []
        for chunk_df in tqdm(iterator, total=num_chunks, desc="Loading data"):
            list_of_dfs.append(chunk_df)

        print("\nConcatenating all loaded chunks...")
        df = pd.concat(list_of_dfs, ignore_index=True)
        print(f"Successfully loaded {len(df)} rows into DataFrame.")

        conn.close()
        print("Database connection closed.")

        print("Processing data types...")
        df['game_id'] = df['game_id'].astype(str)
        df['pctimestring'] = df['pctimestring'].astype(str)
        df['score'] = df['score'].astype(str)
        df['scoremargin'] = df['scoremargin'].astype(str)
        df['eventnum'] = pd.to_numeric(df['eventnum'], errors='coerce')
        df['eventmsgtype'] = pd.to_numeric(df['eventmsgtype'], errors='coerce')
        df['eventmsgactiontype'] = pd.to_numeric(df['eventmsgactiontype'], errors='coerce')
        df['period'] = pd.to_numeric(df['period'], errors='coerce')
        desc_cols = ['homedescription', 'neutraldescription', 'visitordescription']
        for col in desc_cols:
            df[col] = df[col].fillna('')
        print("Data type processing complete.")

        print("\nAttempting to determine game outcomes...")
        game_outcomes = pd.Series(dtype=int)
        end_game_events = df[(df['eventmsgtype'] == 13) & (df['score'].str.contains(' - ', na=False))].copy()

        if not end_game_events.empty:
            end_game_events = end_game_events.dropna(subset=['period'])
            if not end_game_events.empty:
                end_game_events['period'] = end_game_events['period'].astype(int)
                final_events = end_game_events.sort_values('period').groupby('game_id').last()

                if 'score' in final_events.columns:
                    scores_split = final_events['score'].str.split(' - ', expand=True)
                    scores_split.columns = ['home_score', 'visitor_score']
                    scores_split['home_score'] = pd.to_numeric(scores_split['home_score'], errors='coerce')
                    scores_split['visitor_score'] = pd.to_numeric(scores_split['visitor_score'], errors='coerce')
                    scores_split = scores_split.dropna(subset=['home_score', 'visitor_score'])

                    if not scores_split.empty:
                        scores_split['home_win'] = (scores_split['home_score'] > scores_split['visitor_score']).astype(int)
                        game_outcomes = scores_split['home_win']
                        print(f"Determined outcomes for {len(game_outcomes)} games.")
                        if limit_rows:
                            print("[Warning] Game outcomes may be incomplete due to the row limit.")

        print("Merging game outcomes back to the main DataFrame...")
        if not game_outcomes.empty:
             df_with_outcome = df.merge(game_outcomes.rename('home_win'), on='game_id', how='left')
        else:
             print("No game outcomes determined, adding 'home_win' column with NaN.")
             df['home_win'] = pd.NA
             df_with_outcome = df

        print("--- Data loading and processing finished ---")
        return df_with_outcome

    except sqlite3.Error as e:
        print(f"\n--- Database Error ---")
        print(f"An error occurred while interacting with the database: {e}")
        return None
    except FileNotFoundError:
        print(f"\n--- File Not Found Error ---")
        print(f"Error: The database file was not found at the specified path: {db_path}")
        return None
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error details: {e}")
        return None

"""## 3: 特徴量エンジニアリング用関数の定義"""

def parse_time_to_seconds(time_str: str) -> Optional[int]:
    if isinstance(time_str, str) and ':' in time_str:
        try:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds
        except ValueError:
            return None
    return None

def calculate_seconds_elapsed(row: pd.Series) -> Optional[float]:
    period = row['period']
    pctimestring = row['pctimestring']
    if pd.isna(period) or period < 1:
        return None
    seconds_in_period = parse_time_to_seconds(pctimestring)
    if seconds_in_period is None:
        return None
    seconds_per_period = 720 if period <= 4 else 300
    seconds_elapsed_in_current_period = seconds_per_period - seconds_in_period
    if seconds_elapsed_in_current_period < 0 or seconds_elapsed_in_current_period > seconds_per_period:
         return None
    if period <= 4:
        total_seconds_elapsed = (period - 1) * 720 + seconds_elapsed_in_current_period
    else:
        total_seconds_elapsed = 4 * 720 + (period - 5) * 300 + seconds_elapsed_in_current_period
    return total_seconds_elapsed

def process_score_margin(margin_str: str) -> Optional[int]:
    if margin_str == 'TIE':
        return 0
    elif pd.isna(margin_str) or margin_str == '':
         return None
    else:
        try:
            return int(str(margin_str).replace('+', ''))
        except ValueError:
            return None

"""## 4: 設定とデータ読み込みの実行"""
# ご自身の環境に合わせてパスを修正してください
db_file = r'C:\Users\amilu\Projects\vsCodeFile\PBL\nba.sqlite' # 研究室PC用path
# db_file = r'C:\Programing\PBL\nba.sqlite' # ノートPC用path

# 処理する行数を設定 (Noneにすると全データを対象)
# limit_rows = 500000 # 動作確認には50万行程度がおすすめ
#limit_rows = 45616 # 元のコードの値
limit_rows = 225109
# limit_rows = None

df_processed = load_and_process_pbp(db_file, limit_rows=limit_rows)

if df_processed is not None:
    print("\nShape of loaded data:", df_processed.shape)
else:
    print("Data loading failed.")

"""## 5: 特徴量エンジニアリングの実行"""
if df_processed is not None:
    print("\n--- Feature Engineering ---")
    print("Calculating total seconds elapsed...")
    df_processed['seconds_elapsed'] = df_processed.apply(calculate_seconds_elapsed, axis=1)
    print("Processing score margin...")
    df_processed['numeric_score_margin'] = df_processed['scoremargin'].apply(process_score_margin)
    print("Forward filling missing 'numeric_score_margin' within each game...")
    df_processed = df_processed.sort_values(by=['game_id', 'eventnum'])
    df_processed['numeric_score_margin'] = df_processed.groupby('game_id')['numeric_score_margin'].ffill()
    print("Generating composite event ID...")
    df_processed['composite_event_id'] = (df_processed['eventmsgtype'] * 1000 + df_processed['eventmsgactiontype'])
    print("Feature Engineering complete.")

"""## 6: モデル用データ準備 (フィルタリング)"""
if df_processed is not None:
    print("\n--- Data Preparation for Modeling ---")
    initial_rows = len(df_processed)
    model_df = df_processed.dropna(subset=['home_win', 'seconds_elapsed', 'numeric_score_margin', 'period', 'composite_event_id']).copy()
    model_df = model_df[model_df['period'] > 0]
    model_df = model_df[model_df['eventmsgtype'] != 12] # "Start Period" イベントを除外
    filtered_rows = len(model_df)
    print(f"Rows before filtering: {initial_rows}")
    print(f"Rows after filtering invalid/unnecessary entries: {filtered_rows}")
    if filtered_rows == 0:
        print("No data left after filtering. Exiting.")
        model_df = pd.DataFrame() # 空にして以降の処理をスキップ

"""## 7: 特徴量とターゲットの選択、訓練/テスト分割"""
if 'model_df' in locals() and not model_df.empty:
    print("Applying Label Encoding to 'composite_event_id'...")
    le = LabelEncoder()
    model_df['composite_event_id'] = le.fit_transform(model_df['composite_event_id'])
    
    features = ['numeric_score_margin', 'seconds_elapsed', 'composite_event_id']
    target = 'home_win'
    print(f"\nSelected features: {features}")

    X = model_df[features]
    y = model_df[target].astype(int)
    groups = model_df['game_id']

    print("\nSplitting data into training and testing sets (game-aware)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    if groups.nunique() < 2:
        print("Warning: Not enough unique games for GroupShuffleSplit. Using regular split.")
        train_idx, test_idx = train_test_split(range(len(X)), test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
    else:
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]
    
    # 後で予測結果を結合するために、テストデータの元のDataFrameを保持
    model_df_test = model_df.iloc[test_idx].copy()

    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    print(f"Training games: {groups_train.nunique()}")
    print(f"Testing games: {groups_test.nunique()}")
else:
    print("Skipping feature selection and train/test split as model_df is not available or empty.")

"""## 8: LSTM用データへの変換 (スケーリング、パディング) - ★★★修正箇所★★★"""
if 'X_train' in locals() and not X_train.empty:
    print("\n--- Preparing Data for LSTM ---")
    
    # 1. スケーリング
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # スケール後のデータをDataFrameに戻して、game_idと結合
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=features)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=features)
    
    # 2. 試合ごとにグループ化し、シーケンスを作成
    def create_padded_sequences(X_scaled_df: pd.DataFrame, y_series: pd.Series, groups_series: pd.Series) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """試合ごとにグループ化し、シーケンスをパディングしてNumpy配列を返す"""
        game_ids = groups_series.unique()
        X_sequences = [X_scaled_df[groups_series == game_id].values for game_id in game_ids]
        y_sequences = [y_series[groups_series == game_id].values for game_id in game_ids]
        
        # 元のシーケンス長を保持 (評価時にパディングを除外するため)
        original_lengths = [len(seq) for seq in X_sequences]
        
        # パディング処理
        X_padded = pad_sequences(X_sequences, padding='pre', dtype='float32')
        y_padded = pad_sequences(y_sequences, padding='pre', value=-1) # ターゲットは-1でパディング
        
        # ターゲットの形状を (サンプル数, タイムステップ数, 1) に変換
        y_padded = np.expand_dims(y_padded, -1)
        
        return X_padded, y_padded, original_lengths

    print("Creating padded sequences for train and test sets...")
    X_train_scaled_lstm, y_train_lstm, _ = create_padded_sequences(X_train_scaled_df, y_train, groups_train)
    X_test_scaled_lstm, y_test_lstm, test_original_lengths = create_padded_sequences(X_test_scaled_df, y_test, groups_test)
    
    print(f"Shape of X_train for LSTM: {X_train_scaled_lstm.shape}")
    print(f"Shape of y_train for LSTM: {y_train_lstm.shape}")
    print(f"Shape of X_test for LSTM: {X_test_scaled_lstm.shape}")
    print(f"Shape of y_test for LSTM: {y_test_lstm.shape}")
else:
    print("Skipping LSTM data preparation as training data is not available.")


"""## 9: モデルの定義とコンパイル"""
if 'X_train_scaled_lstm' in locals():
    print("\n--- Defining and Compiling Many-to-Many LSTM Model ---")
    model = Sequential([
        LSTM(32, input_shape=(X_train_scaled_lstm.shape[1], X_train_scaled_lstm.shape[2]), return_sequences=True),
        Dropout(0.3),
        TimeDistributed(Dense(16, activation='relu')),
        Dropout(0.3),
        TimeDistributed(Dense(1, activation='sigmoid'))
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
else:
    print("Skipping model definition as LSTM training data is not available.")

"""## 10: モデルの訓練"""
if 'model' in locals():
    print("\n--- Training LSTM Model ---")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train_scaled_lstm, y_train_lstm,
                        epochs=50,
                        batch_size=16,
                        validation_split=0.2,
                        callbacks=[early_stopping])
    print("Model training complete.")
else:
    print("Skipping model training as the model is not defined.")

"""## 11: 予測と評価"""
if 'model' in locals() and 'X_test_scaled_lstm' in locals() and X_test_scaled_lstm.size > 0:
    print("\n--- Prediction and Evaluation with Many-to-Many LSTM ---")
    y_pred_proba_3d = model.predict(X_test_scaled_lstm)

    # パディングを除外して、予測値と真の値をフラットなリストに変換
    y_pred_flat = []
    y_true_flat = []
    for i, length in enumerate(test_original_lengths):
        # シーケンスの末尾から元の長さだけ取得
        valid_preds = y_pred_proba_3d[i, -length:, 0]
        valid_true = y_test_lstm[i, -length:, 0]
        y_pred_flat.extend(valid_preds)
        y_true_flat.extend(valid_true)

    if y_true_flat and y_pred_flat:
        y_pred_class_flat = (np.array(y_pred_flat) > 0.5).astype(int)
        accuracy = accuracy_score(y_true_flat, y_pred_class_flat)
        auc = roc_auc_score(y_true_flat, y_pred_flat)
        logloss = log_loss(y_true_flat, y_pred_flat)
        brier = brier_score_loss(y_true_flat, y_pred_flat)

        print(f"\nLSTM Model Evaluation Results (Per Play):")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  ROC AUC:     {auc:.4f}")
        print(f"  Log Loss:    {logloss:.4f}")
        print(f"  Brier Score: {brier:.4f}")

        # 予測結果を元のテストデータフレームに追加
        if len(y_pred_flat) == len(model_df_test):
            model_df_test['win_probability_pred'] = y_pred_flat
            print("\nFirst 15 rows of test data with time-varying win probability:")
            display_cols = ['game_id', 'eventnum', 'period', 'pctimestring', 'numeric_score_margin', 'home_win', 'win_probability_pred']
            print(model_df_test[display_cols].head(15))
        else:
            print(f"Warning: Length mismatch when adding predictions. Predictions: {len(y_pred_flat)}, Test set: {len(model_df_test)}")
    else:
        print("No data to evaluate.")
else:
    print("Skipping prediction and evaluation as model or test data is not available.")
    model_df_test = pd.DataFrame() # 後続処理のために空で定義

"""## 12: CSV出力"""
if 'model_df_test' in locals() and not model_df_test.empty:
    num_games_to_export = 3
    unique_test_games = model_df_test['game_id'].unique()
    if len(unique_test_games) > 0:
        games_to_export = unique_test_games[:num_games_to_export]
        df_for_export = model_df_test[model_df_test['game_id'].isin(games_to_export)]
        print(f"\nExporting data for {len(games_to_export)} games...")
        output_csv_path = f'lstm_predictions_{len(games_to_export)}_games.csv'
        try:
            df_for_export.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"Successfully exported test predictions to: {output_csv_path}")
        except Exception as e:
            print(f"An error occurred while exporting to CSV: {e}")
    else:
        print("\nNo games found in the test set to export.")
else:
    print("\nSkipping CSV export because the final test DataFrame ('model_df_test') is not available or is empty.")

"""## 13: モメンタムの取り出し (WPA分析)"""
if 'model_df_test' in locals() and not model_df_test.empty and 'le' in locals():
    print("\n--- Calculating Win Probability Added (WPA) for each event ---")
    
    event_id_to_name_map = {
        # --- 基本カテゴリ（データに存在しないIDのフォールバック用） ---
        1000: 'FG成功(その他)', 
        2000: 'FG失敗(その他)', 
        3000: 'フリースロー(その他)', 
        4000: 'リバウンド',
        5000: 'ターンオーバー(その他)', 
        6000: 'ファウル(その他)', 
        7000: 'バイオレーション', 
        8000: '選手交代',
        9000: 'タイムアウト', 
        10000: 'ジャンプボール', 
        11000: '退場', 
        12000: 'ピリオド開始',
        13000: 'ピリオド終了', 
        18000: 'その他',

        # --- 詳細アクション（今回の分析結果を反映） ---
        # ショット成功 (Type 1)
        1001: 'ジャンプショット成功',
        1002: 'ランニング/フローター成功',  # 分析により追加
        1003: 'スラムダンク成功',         # 分析により追加
        1005: 'レイアップ成功',
        1006: 'フックショット成功',       # 分析により追加
        1007: 'ティップイン/アリウープ成功', # 分析により追加

        # ショット失敗 (Type 2)
        2001: 'ジャンプショット失敗',
        2002: 'ランニング系ショット失敗',   # 分析により追加
        2005: 'レイアップ失敗',
        2006: 'フックショット失敗',       # 分析により追加

        # フリースロー (Type 3)
        3010: 'フリースロー成功 (1st/2nd)', # 表現を更新
        3011: 'フリースロー成功 (2nd/3rd)', # 表現を更新
        3012: 'フリースロー失敗',
        3013: 'テクニカルフリースロー成功', # 分析により追加

        # ターンオーバー (Type 5)
        5001: 'スティール',              # 分析により追加
        5002: 'ボールロスト/パスミス',     # 表現を更新
        5004: 'オフェンスファウル/バイオレーション', # 表現を更新

        # ファウル (Type 6)
        6001: 'パーソナルファウル',
        6002: 'シューティングファウル',
        6003: 'オフェンスチャージングファウル' # 分析により更新
    }

    # LabelEncoderのエンコード後の値(0,1,2...)から元の複合ID(1001, 1005...)を取得
    encoded_label_to_original_id_map = {i: original_id for i, original_id in enumerate(le.classes_)}
    # エンコード後の値からイベント名へのマッピング辞書を作成
    encoded_label_to_name_map = {
        encoded_label: event_id_to_name_map.get(original_id, f'不明なID({original_id})')
        for encoded_label, original_id in encoded_label_to_original_id_map.items()
    }

    wpa_df = model_df_test.copy()
    wpa_df['win_prob_before'] = wpa_df.groupby('game_id')['win_probability_pred'].shift(1)
    wpa_df['wpa_impact'] = wpa_df['win_probability_pred'] - wpa_df['win_prob_before']
    wpa_df_final = wpa_df.dropna(subset=['wpa_impact', 'composite_event_id'])

    event_impact = wpa_df_final.groupby('composite_event_id')['wpa_impact'].agg(['mean', 'count'])
    event_impact['event_name'] = event_impact.index.map(encoded_label_to_name_map)
    event_impact = event_impact.sort_values(by='mean', ascending=False)

    print("\n--- 詳細なイベントごとのWPAインパクト Top15 ---")
    print("\n[勝率を最も高めたイベント (ホームチーム)]")
    print(event_impact.head(15))
    print("\n[勝率を最も下げたイベント (ホームチーム)]")
    print(event_impact.tail(15).sort_values(by='mean'))

    event_impact_filtered = event_impact[event_impact['count'] >= 5]
    if not event_impact_filtered.empty:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        fig.suptitle('詳細イベントごとの平均WPAインパクト (発生回数5回以上)', fontsize=18)
        top_events = event_impact_filtered.head(15)
        sns.barplot(ax=axes[0], x=top_events['mean'], y=top_events['event_name'], palette='Greens_r')
        axes[0].set_title('勝率を上げるイベント Top 15')
        axes[0].set_xlabel('平均WPAインパクト (勝率上昇)')
        axes[0].set_ylabel('イベント名')
        bottom_events = event_impact_filtered.tail(15).sort_values(by='mean', ascending=True)
        sns.barplot(ax=axes[1], x=bottom_events['mean'], y=bottom_events['event_name'], palette='Reds')
        axes[1].set_title('勝率を下げるイベント Top 15')
        axes[1].set_xlabel('平均WPAインパクト (勝率下降)')
        axes[1].set_ylabel('')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    else:
        print("\nNo events found with count >= 5 for plotting.")
else:
    print("\nSkipping WPA analysis because the final test DataFrame ('model_df_test') is not available or is empty.")

print("\n--- Win probability prediction notebook execution finished ---")