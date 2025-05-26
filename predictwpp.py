import sqlite3
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
# from sklearn.linear_model import LogisticRegression # 不要になるためコメントアウト
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss, classification_report
import numpy as np
from typing import Optional
import os
from tqdm.notebook import tqdm

# --- Keras/TensorFlow のインポート ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

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
        # 1. SQLiteデータベースへの接続
        print("Connecting to database...")
        conn = sqlite3.connect(db_path)
        print("Database connection successful.")

        # --- プログレスバー用に全体の行数を先に取得 ---
        total_rows_to_load = limit_rows
        if total_rows_to_load is None:
            print("Determining total row count for progress bar...")
            # limitがない場合はテーブルの全行数を取得
            count_query = "SELECT COUNT(*) FROM play_by_play;"
            total_rows_to_load = pd.read_sql_query(count_query, conn).iloc[0, 0]
            print(f"Total rows in table: {total_rows_to_load}")

        # 2. SQLクエリの構築
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

        # --- チャンクでデータを読み込み、プログレスバーを表示 ---
        chunk_size = 100000  # 一度に読み込む行数（この値はPCのメモリに応じて調整可能）
        print(f"Loading data in chunks of {chunk_size} rows...")

        # chunksizeを指定すると、DataFrameのイテレータ（分割されたデータ）が返る
        iterator = pd.read_sql_query(query, conn, chunksize=chunk_size)

        # tqdmでプログレスバーを設定（total=チャンクの総数）
        num_chunks = (total_rows_to_load + chunk_size - 1) // chunk_size

        list_of_dfs = []
        # イテレータをtqdmでラップすると、ループの進捗が表示される
        for chunk_df in tqdm(iterator, total=num_chunks, desc="Loading data"):
            list_of_dfs.append(chunk_df)

        # 読み込んだ全てのチャンクを一つのDataFrameに結合
        print("\nConcatenating all loaded chunks...")
        df = pd.concat(list_of_dfs, ignore_index=True)
        print(f"Successfully loaded {len(df)} rows into DataFrame.")

        # 4. データベース接続を閉じる
        conn.close()
        print("Database connection closed.")

        # 5. データ型の確認と変換（ここから先の処理は元のコードと同じ）
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

        # --- 最終スコアの取得と勝敗判定 ---
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

def parse_time_to_seconds(time_str):
    """ 'MM:SS' 形式の文字列を秒に変換 """
    if isinstance(time_str, str) and ':' in time_str:
        try:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds
        except ValueError:
            return None # パースエラー
    return None

def calculate_seconds_elapsed(row):
    """ 試合開始からの経過秒数を計算 """
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

def process_score_margin(margin_str):
    """ scoremargin を数値に変換 ('TIE' -> 0) """
    if margin_str == 'TIE':
        return 0
    elif pd.isna(margin_str) or margin_str == '':
         return None
    else:
        try:
            return int(str(margin_str).replace('+', ''))
        except ValueError:
            return None

"""## 4 設定とデータ読み込みの実行"""

#研究室PC用path
#db_file = r'C:\Users\amilu\Projects\vsCodeFile\PBL\nba.sqlite'
#ノートPC用path
db_file = r'C:\Programing\PBL\nba.sqlite'

limit_rows = 45616 #9999808
#45616
# limit_rows = None # 全データの場合

df_processed = load_and_process_pbp(db_file, limit_rows=limit_rows)

# 読み込み結果の確認
if df_processed is not None:
    print("\nShape of loaded data:", df_processed.shape)
    df_processed.head() # 先頭数行を表示
else:
    print("Data loading failed.")

"""## 5: 特徴量エンジニアリングの実行"""

if df_processed is not None:
    print("\n--- Feature Engineering ---")

    print("Calculating total seconds elapsed...")
    df_processed['seconds_elapsed'] = df_processed.apply(calculate_seconds_elapsed, axis=1)

    print("Processing score margin...")
    # Step 1: 'TIE' や数値変換可能なものを数値に変換 (NoneはNoneのまま)
    df_processed['numeric_score_margin'] = df_processed['scoremargin'].apply(process_score_margin)

    # Step 2: game_id ごとに並べ替え、欠損値 (None) を直前の有効な値で前方補完 (ffill)
    print("Forward filling missing 'numeric_score_margin' within each game...")
    # game_id と eventnum でソートすることが重要
    df_processed = df_processed.sort_values(by=['game_id', 'eventnum'])
    df_processed['numeric_score_margin'] = df_processed.groupby('game_id')['numeric_score_margin'].ffill()

    # ffill 後も NaN が残る場合がある（ゲームの最初の数プレイなど、前に有効な値がない場合）
    # これらは後続の dropna で処理されるか、別途 0 などで埋める判断も可能
    # print("NaN count in numeric_score_margin after ffill:", df_processed['numeric_score_margin'].isnull().sum())
    # 2.3 Generate Composite Event ID
    print("Generating composite event ID...")
    # eventmsgtype と eventmsgactiontype が数値であることを確認 (NaNの場合は計算結果もNaNになる)
    # 複合IDの計算前に、これらのカラムの欠損値処理が必要か検討
    # 例: df_processed['eventmsgtype'].fillna(0, inplace=True) # 欠損を0で埋める場合
    #     df_processed['eventmsgactiontype'].fillna(0, inplace=True)

    # もし欠損値のまま計算すると、結果がNaNになるため、後続のdropnaで除外されるか、
    # またはfillna(例えば -1 や特定の予約ID) で埋める
    df_processed['composite_event_id'] = (df_processed['eventmsgtype'] * 1000 + df_processed['eventmsgactiontype'])
    # 欠損値から生じたNaNを、例えば不明なID (-1など) で埋める場合
    # df_processed['composite_event_id'] = df_processed['composite_event_id'].fillna(-1).astype(int)
    print("Composite event ID generation complete.")


    # 結果の確認
    print("\nPreview of processed time, score margin, and composite event ID features:")
    print(df_processed[['game_id', 'eventnum', 'eventmsgtype', 'eventmsgactiontype', 'composite_event_id', 'seconds_elapsed', 'numeric_score_margin']].head(15))

"""## 6: モデル用データ準備 (フィルタリング)"""

if df_processed is not None:
    print("\n--- Data Preparation for Modeling ---")
    print("Filtering data for modeling...")
    initial_rows = len(df_processed)

    # 欠損値の確認 (フィルタリング前)
    # print("NaN counts before filtering:")
    # print(df_processed[['home_win', 'seconds_elapsed', 'numeric_score_margin', 'period']].isnull().sum())

    model_df = df_processed.dropna(subset=['home_win', 'seconds_elapsed', 'numeric_score_margin', 'period', 'composite_event_id'])
    model_df = model_df[model_df['period'] > 0]
    model_df = model_df[model_df['eventmsgtype'] != 12] # "Start Period" イベントを除外

    filtered_rows = len(model_df)
    print(f"Rows before filtering: {initial_rows}")
    print(f"Rows after filtering invalid/unnecessary entries: {filtered_rows}")

    if filtered_rows > 0:
        model_df.head()
    else:
        print("No data left after filtering.")

"""## 7: 特徴量とターゲットの選択、訓練/テスト分割,LSTM準備"""

if 'model_df' in locals() and not model_df.empty:
    # --- Label Encoding for composite_event_id ---
    print("Applying Label Encoding to 'composite_event_id'...")
    le = LabelEncoder() # LabelEncoderのインスタンスをここで作成
    # SettingWithCopyWarning を回避するために .copy() を使用
    model_df_encoded = model_df.copy() # この変数名でエンコード済みDataFrameを作成
    model_df_encoded['composite_event_id'] = le.fit_transform(model_df_encoded['composite_event_id'])
    print(f"Shape after Label Encoding: {model_df_encoded.shape}")


    # --- 特徴量とターゲットの選択 ---
    # ラベルエンコードされた 'composite_event_id' を特徴量に追加
    features = ['numeric_score_margin', 'seconds_elapsed', 'composite_event_id']
    target = 'home_win'

    print(f"\nTotal features for the model: {len(features)}")
    print(f"Selected features: {features}")

    X = model_df_encoded[features]
    y = model_df_encoded[target].astype(int)

    print("\nSplitting data into training and testing sets (game-aware)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # グループ分割のためにユニークなgame_idが2つ以上必要
    if model_df_encoded['game_id'].nunique() < 2:
        print("Warning: Not enough unique games for GroupShuffleSplit. Falling back to regular split or adjusting test_size if possible.")
        if len(X) > 1:
             train_idx, test_idx = train_test_split(range(len(X)), test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
        else: # サンプルが少なすぎる場合
            train_idx, test_idx = range(len(X)), [] # 全て訓練データとするか、エラー処理
            print("Too few samples for splitting. Using all as training data if any.")
    else:
        train_idx, test_idx = next(gss.split(X, y, groups=model_df_encoded['game_id']))

    if len(train_idx) > 0 and len(test_idx) > 0:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")

    elif len(train_idx) > 0: # テストデータが作成できなかった場合
        print(f"Only training data available. Training set size: {len(X_train)}")
        # X_test, y_test を空のDataFrame/Seriesとして初期化
        X_test, y_test = pd.DataFrame(columns=X.columns), pd.Series(dtype=y.dtype)
    else: # 訓練データもテストデータも作成できなかった場合
        print("Could not create valid train/test splits. Insufficient data or groups.")
        # X_train, X_test, y_train, y_test を適切に処理 (例: 空のデータフレーム)
        X_train, X_test = pd.DataFrame(columns=X.columns), pd.DataFrame(columns=X.columns)
        y_train, y_test = pd.Series(dtype=y.dtype), pd.Series(dtype=y.dtype)
else:
    print("Skipping feature selection and train/test split as model_df is not available or empty.")

"""##  10: モデルの定義とコンパイル"""

if 'X_train_scaled_lstm' in locals():
    print("\n--- Defining and Compiling Many-to-Many LSTM Model ---")

    model = Sequential([
        # return_sequences=True に変更して、各タイムステップの出力を次に渡す
        LSTM(32, input_shape=(X_train_scaled_lstm.shape[1], X_train_scaled_lstm.shape[2]), return_sequences=True),
        Dropout(0.3),
        # TimeDistributedを使って、各タイムステップの出力にDenseレイヤーを適用
        TimeDistributed(Dense(16, activation='relu')),
        Dropout(0.3),
        TimeDistributed(Dense(1, activation='sigmoid'))
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
else:
    print("Skipping model definition as LSTM training data is not available.")

"""## 11: モデルの訓練"""

if 'model' in locals():
    print("\n--- Training LSTM Model ---")

    # 過学習を防ぐためのEarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train_scaled_lstm, y_train_lstm,
                        epochs=50,
                        batch_size=16,
                        validation_split=0.2, # 訓練データの一部を検証用に使う
                        callbacks=[early_stopping])

    print("Model training complete.")
else:
    print("Skipping model training as the model is not defined.")

"""## 12:予測と評価"""

if 'model' in locals() and 'X_test_scaled_lstm' in locals() and X_test_scaled_lstm.size > 0 : # X_test_scaled_lstmが空でないことも確認
    print("\n--- Prediction and Evaluation with Many-to-Many LSTM ---")

    y_pred_proba_3d = model.predict(X_test_scaled_lstm)

    # test_indicesからテスト対象のgame_idを取得
    # groupedはセル[9] (id: 92940c3b)で定義されている前提
    # train_indices, test_indicesもセル[9]で定義されている前提
    all_game_ids_in_grouped = np.array([name for name, group in grouped])

    if len(all_game_ids_in_grouped) > 0 and len(test_indices) > 0 and max(test_indices) < len(all_game_ids_in_grouped):
        test_game_ids = all_game_ids_in_grouped[test_indices]
    else:
        test_game_ids = np.array([]) # 安全策
        print("Warning: Could not determine test_game_ids reliably.")

    # 元のテストシーケンスの長さを取得
    original_lengths = []
    if len(test_game_ids) > 0:
         original_lengths = [len(group) for name, group in grouped if name in test_game_ids]

    y_pred_flat = []
    y_true_flat = []

    if len(original_lengths) == y_pred_proba_3d.shape[0]: # 形状が一致する場合のみ処理
        for i, length in enumerate(original_lengths):
            valid_preds = y_pred_proba_3d[i, -length:, 0]
            y_pred_flat.extend(valid_preds)

            valid_true = y_test_lstm[i, -length:, 0] # y_test_lstmもセル[9]で定義済み
            y_true_flat.extend(valid_true)
    else:
        print("Warning: Mismatch between number of original_lengths and predictions. Evaluation might be incorrect.")


    if y_true_flat and y_pred_flat: # データがある場合のみ評価
        y_pred_class_flat = (np.array(y_pred_flat) > 0.5).astype(int)
        accuracy = accuracy_score(y_true_flat, y_pred_class_flat)
        auc = roc_auc_score(y_true_flat, y_pred_flat)
        logloss = log_loss(y_true_flat, y_pred_flat)
        brier = brier_score_loss(y_true_flat, y_pred_flat)

        print(f"LSTM Model Evaluation Results (Per Play):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC AUC:  {auc:.4f}")
        print(f"  Log Loss: {logloss:.4f}")
        print(f"  Brier Score: {brier:.4f}")
    else:
        print("No data to evaluate.")

    # --- 予測結果を元のテストデータに戻す ---
    # model_df_encoded (LabelEncodedされたデータ) から test_idx (セル[8]で定義)を使ってテストデータを再構築
    # X_test (セル[8]で定義) のインデックスを利用する方が安全
    if not X_test.empty:
        model_df_test = model_df_encoded.loc[X_test.index].copy()
        if len(y_pred_flat) == len(model_df_test):
             model_df_test['win_probability_pred'] = y_pred_flat
             print("\nFirst 15 rows of test data with time-varying win probability:")
             display_cols = ['game_id', 'eventnum', 'period', 'pctimestring', 'numeric_score_margin', 'home_win', 'win_probability_pred']
             print(model_df_test[display_cols].head(15))
        else:
            print("Warning: Length mismatch when adding predictions to model_df_test.")
            # model_df_test は存在するが予測結果を結合できない状態
            if 'win_probability_pred' in model_df_test.columns:
                 model_df_test.drop(columns=['win_probability_pred'], inplace=True) # 不完全な列を削除
    else:
        print("X_test is empty, cannot create model_df_test with predictions.")
        model_df_test = pd.DataFrame() # 空のDataFrameとして定義

else:
    print("Skipping prediction and evaluation as model or test data is not available or X_test_scaled_lstm is empty.")
    model_df_test = pd.DataFrame() # model_df_testを空で定義しておく

"""## csv出力"""

# 最終的なテストデータフレームが存在し、空でないことを確認
if 'model_df_test' in locals() and isinstance(model_df_test, pd.DataFrame) and not model_df_test.empty:

    # --- ここでCSVに出力する試合数を設定 ---
    num_games_to_export = 3  # 例として3試合に設定（この数値を自由に変更してください）

    # テストデータに含まれるユニークなゲームIDを取得
    unique_test_games = model_df_test['game_id'].unique()

    if len(unique_test_games) > 0:
        # 出力するゲームIDを、設定した数だけスライスして取得
        games_to_export = unique_test_games[:num_games_to_export]

        # 選択されたゲームIDのデータのみを新しいDataFrameに抽出
        df_for_export = model_df_test[model_df_test['game_id'].isin(games_to_export)]

        print(f"\nExporting data for {len(games_to_export)} games (out of {len(unique_test_games)} total test games).")

        # 出力ファイル名を分かりやすく動的に設定
        output_csv_path = f'lstm_predictions_{len(games_to_export)}_games.csv'
        print(f"--- Exporting Final Test Predictions to CSVA ---")

        try:
            # 抽出したデータフレーム（df_for_export）をCSVファイルに保存
            df_for_export.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"Successfully exported test predictions to: {output_csv_path}")

        except Exception as e:
            print(f"An error occurred while exporting to CSV: {e}")

    else:
        print("\nNo games found in the test set to export.")

else:
    print("\nSkipping CSV export because the final test DataFrame ('model_df_test') was not found or is empty.")


print("\n--- Win probability prediction notebook execution finished ---")

"""## モメンタムの取り出し"""

import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

# 分析対象のデータフレーム (model_df_test) と LabelEncoder (le) が存在することを確認
if 'model_df_test' in locals() and isinstance(model_df_test, pd.DataFrame) and \
   'le' in locals() and isinstance(le, LabelEncoder): # le (LabelEncoderのインスタンス) もチェック

    print("--- Calculating Win Probability Added (WPA) for each event ---")

    # 1. 元の複合IDからイベント名へのマッピング辞書を定義 (変更なし)
    event_id_to_name_map = {
        1000: 'FG成功(その他)', 2000: 'FG失敗(その他)', 3000: 'フリースロー', 4000: 'リバウンド',
        5000: 'ターンオーバー(その他)', 6000: 'ファウル(その他)', 7000: 'バイオレーション', 8000: '選手交代',
        9000: 'タイムアウト', 10000: 'ジャンプボール', 11000: '退場', 12000: 'ピリオド開始',
        13000: 'ピリオド終了', 18000: 'その他',
        1001: 'ジャンプショット成功', 1005: 'レイアップ成功', 1007: 'ダンク成功',
        1041: 'アリウープ成功', 1042: 'ドライビングレイアップ成功', 1050: 'ランニングダンク成功',
        1052: 'アリウープダンク成功', 1055: 'フックショット成功', 1066: 'ジャンプバンクショット成功',
        1108: 'カッティングダンクショット成功',
        2001: 'ジャンプショット失敗', 2005: 'レイアップ失敗', 2007: 'ダンク失敗',
        3010: 'フリースロー1本目成功', 3011: 'フリースロー2本目成功', 3012: 'フリースロー失敗',
        5002: 'ボールロスト', 5004: 'トラベリング/オフェンシブファウル', 5040: 'アウトオブバウンズ',
        6001: 'パーソナルファウル', 6002: 'シューティングファウル', 6003: 'テクニカルファウル'
    }

    # ===【重要】ここがマッピングのポイントです ===
    # model_df_test には LabelEncode された composite_event_id が含まれています。
    # LabelEncoder（le）の le.classes_ を使って、
    # エンコード後の連番 (0, 1, 2...) から元の複合IDを経由してイベント名へのマッピング辞書を作成します。

    # encoded_label (0,1,2...) -> original_composite_id (1001, 1005...)
    encoded_label_to_original_id_map = {i: original_id for i, original_id in enumerate(le.classes_)}

    # encoded_label (0,1,2...) -> event_name ("ジャンプショット成功", "レイアップ成功"...)
    encoded_label_to_name_map = {
        encoded_label: event_id_to_name_map.get(original_id, f'不明なID({original_id})')
        for encoded_label, original_id in encoded_label_to_original_id_map.items()
    }

    # 3. WPAインパクトを計算 (対象は model_df_test)
    # model_df_test はセル[11] (id:d00b13e1)で予測結果が追加されている前提
    wpa_df = model_df_test.copy() # 以降の処理で元のmodel_df_testを変更しないようにコピー
    wpa_df['win_prob_before'] = wpa_df.groupby('game_id')['win_probability_pred'].shift(1)
    wpa_df['wpa_impact'] = wpa_df['win_probability_pred'] - wpa_df['win_prob_before']

    # WPAインパクト計算に必要なデータが揃っている行のみを対象にする
    wpa_df_final = wpa_df.dropna(subset=['wpa_impact', 'composite_event_id']) # composite_event_idはエンコード済みのもの

    # 4. エンコード後の composite_event_id ごとにWPAインパクトの平均と発生回数を計算
    event_impact = wpa_df_final.groupby('composite_event_id')['wpa_impact'].agg(['mean', 'count'])

    # 新しく作成した「エンコード後の連番 -> イベント名」のマッピング辞書を使ってイベント名を追加
    event_impact['event_name'] = event_impact.index.map(encoded_label_to_name_map)
    event_impact = event_impact.sort_values(by='mean', ascending=False)

    print("--- 詳細なイベントごとのWPAインパクト Top15 ---")
    print("\n[勝率を最も高めたイベント (ホームチーム)]")
    print(event_impact.head(15))

    print("\n[勝率を最も下げたイベント (ホームチーム)]")
    print(event_impact.tail(15).sort_values(by='mean'))

    # 5. 結果の可視化 (変更なし)
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
    print("Cannot perform WPA analysis because 'model_df_test' or LabelEncoder 'le' is not available or correctly defined.")