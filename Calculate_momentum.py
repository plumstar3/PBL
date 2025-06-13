import sqlite3
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss, classification_report
import numpy as np
from typing import Optional, Tuple, List
import os
from tqdm import tqdm

import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

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
    """
    AA:BBという表記を秒単位の時間に変換
    出力：時刻が秒に変換された値
    """
    if isinstance(time_str, str) and ':' in time_str:
        try:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds
        except ValueError:
            return None
    return None

def calculate_seconds_elapsed(row: pd.Series) -> Optional[float]:
    """
    試合開始からの経過時間(秒)を計算
    出力：試合開始からの経過時間
    """
    period = row['period']
    pctimestring = row['pctimestring']
    if pd.isna(period) or period < 1:
        return None
    #seconds_in_periodに秒単位の時間を格納
    seconds_in_period = parse_time_to_seconds(pctimestring)
    if seconds_in_period is None:
        return None
    #seconds_per_periodにはピリオドの総合時間(秒)を格納。延長の場合は300sを格納。
    seconds_per_period = 720 if period <= 4 else 300
    #seconds_elapsed_in_current_periodにそのピリオド内での経過時間を格納。
    seconds_elapsed_in_current_period = seconds_per_period - seconds_in_period
    #total_seconds_elapsedに試合開始からの経過時間を格納。
    if seconds_elapsed_in_current_period < 0 or seconds_elapsed_in_current_period > seconds_per_period:
        return None
    if period <= 4:
        total_seconds_elapsed = (period - 1) * 720 + seconds_elapsed_in_current_period
    else:
        total_seconds_elapsed = 4 * 720 + (period - 5) * 300 + seconds_elapsed_in_current_period
    return total_seconds_elapsed

def process_score_margin(margin_str: str) -> Optional[int]:
    """
    得点差がTIEならその部分に0を、+3等の表記は+を取り除く
    """
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
#db_file = r'C:\Users\amilu\Projects\vsCodeFile\PBL\nba.sqlite' # 研究室PC用path
#db_file = r'C:\Programing\PBL\nba.sqlite' # ノートPC用path
db_file = 'nba.sqlite'

# 処理する行数を設定 (Noneにすると全データを対象)
# limit_rows = 500000 # 動作確認には50万行程度がおすすめ
limit_rows = 225109 # 元のコードの値
# limit_rows = None

# 複合IDをイベント名に変換
event_id_to_name_map = {
    1000: 'FG成功(その他)',
    1001: 'ジャンプショット成功',
    1005: 'レイアップ成功',
    1007: 'ダンク成功',
    1041: 'アリウープ成功',
    1042: 'ドライビングレイアップ成功',
    1050: 'ランニングダンク成功',
    1052: 'アリウープダンク成功',
    1055: 'フックショット成功',
    1066: 'ジャンプバンクショット成功',
    1108: 'カッティングダンクショット成功',
    2000: 'FG失敗(その他)',
    2001: 'ジャンプショット失敗',
    2002: 'プルアップジャンプショット失敗', # 追加
    2003: 'ドライビングレイアップ失敗',  # 追加
    2004: 'ドライビングダンク失敗',      # 追加
    2005: 'レイアップ失敗',
    2006: 'ランニングレイアップ失敗',    # 追加
    2007: 'ダンク失敗',
    2009: 'フックショット失敗',        # 追加
    3000: 'フリースロー',
    3010: 'フリースロー1本目成功',
    3011: 'フリースロー2本目成功',
    3012: 'フリースロー失敗',
    3013: 'FT 1/2 失敗',             # 追加
    3014: 'FT 2/2 失敗',             # 追加
    3015: 'FT 1/3 失敗',             # 追加
    3017: 'FT 3/3 失敗',             # 追加
    4000: 'リバウンド',
    5000: 'ターンオーバー(その他)',
    5001: 'パスミス ターンオーバー',     # 追加
    5002: 'ボールロスト',
    5003: 'ボールロスト ターンオーバー', # 追加
    5004: 'トラベリング/オフェンシブファウル',
    5007: 'ショットクロック ターンオーバー', # 追加
    5011: 'オフェンス・ゴールテンディング', # 追加
    5040: 'アウトオブバウンズ',
    6000: 'ファウル(その他)',
    6001: 'パーソナルファウル',
    6002: 'シューティングファウル',
    6003: 'テクニカルファウル',
    6004: 'ルーズボールファウル',       # 追加
    6005: 'インバウンドファウル',       # 追加
    6009: 'フレグラントファウル1',     # 追加
    6011: 'オフェンスチャージファウル',  # 追加
    7000: 'バイオレーション',
    7001: 'ダブルドリブル',            # 追加
    7002: 'ドリブル中断バイオレーション',# 追加
    7004: 'キックボール',              # 追加
    8000: '選手交代',
    9000: 'タイムアウト',
    9001: '通常タイムアウト',          # 追加
    9002: 'ショートタイムアウト',        # 追加
    9003: 'オフィシャルタイムアウト',    # 追加
    10000: 'ジャンプボール',
    11000: '退場',
    12000: 'ピリオド開始',
    13000: 'ピリオド終了',
    18000: 'その他',
    1002: 'ドライビングジャンプショット成功',
    1003: 'ドライビングフックショット成功',
    1004: 'ターニングジャンプショット成功',
    1006: 'ステップバックジャンプショット成功',
    1008: 'フローティングジャンプショット成功',
    1009: 'プルアップバンクショット成功',
    2008: 'ドライビングフックショット失敗',
    3016: 'FT 2/3 成功',
    3018: 'FT 3/3 成功',
    3019: 'テクニカルFT成功',
    4001: 'オフェンスリバウンド', # これは後で文脈で判断
    5005: 'オフェンスファウル ターンオーバー',
    5006: 'バッドパス アウトオブバウンズ',
    5008: 'ステップアウトオブバウンズ ターンオーバー',
    5009: 'ボールロスト アウトオブバウンズ',
    5012: 'レーンバイオレーション ターンオーバー',
    5013: 'パーソナルファウル ターンオーバー',
    5014: 'ダブルパーソナル ターンオーバー',
    5015: 'ファウル ターンオーバー',
    5017: 'キックボール ターンオーバー',
    6006: 'アウェイチャレンジ', # 不正確かも
    6009: 'フレグラントファウル1',
    6011: 'オフェンスチャージファウル',
    6013: 'パーソナルブロックファウル',
    6014: 'パーソナルテイクファウル',
    6017: 'シューティングブロックファウル',
    6018: 'ホームチャレンジ', # 不正確かも
    7003: 'ジャンプボールバイオレーション',
    7004: 'キックボールバイオレーション',
    7005: 'レーンバイオレーション',
    7006: 'ディフェンシブゴールテンディング',
    9004: 'チャレンジによるタイムアウト',
    11001: '退場(ファウルアウト)',
    11004: '退場(テクニカルファウル)',
    }

df_processed = load_and_process_pbp(db_file, limit_rows=limit_rows)

if df_processed is not None:
    print("\nShape of loaded data:", df_processed.shape)
else:
    print("Data loading failed.")

"""## 5: 特徴量エンジニアリングの実行
numeric_score_margin列に数値化した得点差を格納
seconds_elapsed列に試合開始からの総経過時間を格納
composite_event_id列に複合IDを格納
"""
def assign_contextual_event_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    descriptionの内容を解析し、イベントの真の主体を判断して
    home_event_id と visitor_event_id を割り当てる。
    """
    print("文脈を考慮したイベントIDの割り当てを開始...")

    # 事前にdescriptionを文字列と大文字に変換
    home_desc = df['homedescription'].fillna('').str.upper()
    visit_desc = df['visitordescription'].fillna('').str.upper()
    base_event_id = (df['eventmsgtype'] * 1000 + df['eventmsgactiontype']).fillna(0).astype(int)

    # 結果を格納する新しい列を0で初期化
    home_ids = pd.Series(0, index=df.index)
    visitor_ids = pd.Series(0, index=df.index)

    # 1. シュート成功 (eventmsgtype=1), シュート失敗 (eventmsgtype=2), フリースロー (eventmsgtype=3)
    # これらは通常、主体が明確
    is_shot_or_ft = df['eventmsgtype'].isin([1, 2, 3])
    home_action_shot = is_shot_or_ft & (home_desc != '') & (visit_desc == '')
    visitor_action_shot = is_shot_or_ft & (visit_desc != '') & (home_desc == '')
    home_ids.loc[home_action_shot] = base_event_id[home_action_shot]
    visitor_ids.loc[visitor_action_shot] = base_event_id[visitor_action_shot]

    # 2. リバウンド (eventmsgtype=4) - "REBOUND" を含む側が主体
    is_rebound = df['eventmsgtype'] == 4
    home_rebound = is_rebound & home_desc.str.contains("REBOUND")
    visitor_rebound = is_rebound & visit_desc.str.contains("REBOUND")
    home_ids.loc[home_rebound] = base_event_id[home_rebound]
    visitor_ids.loc[visitor_rebound] = base_event_id[visitor_rebound]

    # 3. ターンオーバー (eventmsgtype=5) - "STEAL" をした側がポジティブなアクション
    is_turnover = df['eventmsgtype'] == 5
    home_steal = is_turnover & home_desc.str.contains("STEAL") # ホームがスティール -> アウェイのTO
    visitor_steal = is_turnover & visit_desc.str.contains("STEAL") # アウェイがスティール -> ホームのTO
    visitor_ids.loc[home_steal] = base_event_id[home_steal] # アウェイチームのイベントとして記録
    home_ids.loc[visitor_steal] = base_event_id[visitor_steal] # ホームチームのイベントとして記録

    # スティール以外のターンオーバー (例: Lost Ball, Traveling)
    # "TURNOVER" という単語が記述されている側がミスをしたと判断
    no_steal_turnover = is_turnover & ~home_steal & ~visitor_steal
    home_turnover_no_steal = no_steal_turnover & home_desc.str.contains("TURNOVER")
    visitor_turnover_no_steal = no_steal_turnover & visit_desc.str.contains("TURNOVER")
    home_ids.loc[home_turnover_no_steal] = base_event_id[home_turnover_no_steal]
    visitor_ids.loc[visitor_turnover_no_steal] = base_event_id[visitor_turnover_no_steal]

    # 4. ファウル (eventmsgtype=6) - "FOUL" を含む側が主体
    is_foul = df['eventmsgtype'] == 6
    home_foul = is_foul & home_desc.str.contains("FOUL")
    visitor_foul = is_foul & visit_desc.str.contains("FOUL")
    home_ids.loc[home_foul] = base_event_id[home_foul]
    visitor_ids.loc[visitor_foul] = base_event_id[visitor_foul]

    # 5. その他のイベント (タイムアウト、交代など) - 主体が明確な場合が多い
    is_other_event = ~df['eventmsgtype'].isin([1,2,3,4,5,6])
    home_other = is_other_event & (home_desc != '')
    visitor_other = is_other_event & (visit_desc != '')
    home_ids.loc[home_other] = base_event_id[home_other]
    visitor_ids.loc[visitor_other] = base_event_id[visitor_other]
    
    # 割り当てられなかったイベントで片方しか記述がないものは、そちらのイベントとする
    unassigned = (home_ids == 0) & (visitor_ids == 0)
    home_ids.loc[unassigned & (home_desc != '') & (visit_desc == '')] = base_event_id[unassigned & (home_desc != '') & (visit_desc == '')]
    visitor_ids.loc[unassigned & (visit_desc != '') & (home_desc == '')] = base_event_id[unassigned & (visit_desc != '') & (home_desc == '')]


    df['home_event_id'] = home_ids
    df['visitor_event_id'] = visitor_ids
    df['composite_event_id'] = base_event_id # 元のIDも保持
    
    print("イベントIDの割り当て完了。")
    return df

if df_processed is not None:
    print("\n--- 特徴量エンジニアリング ---")
    print("総経過時間を計算中")
    df_processed['seconds_elapsed'] = df_processed.apply(calculate_seconds_elapsed, axis=1)
    print("得点差処理中")
    df_processed['numeric_score_margin'] = df_processed['scoremargin'].apply(process_score_margin)
    print("numeriC_score_marginをマッチング")
    df_processed = df_processed.sort_values(by=['game_id', 'eventnum'])
    df_processed['numeric_score_margin'] = df_processed.groupby('game_id')['numeric_score_margin'].ffill()

    # visitor - home を home - visitor に変換
    print("得点差の正負を標準的な表記 (home - visitor) に修正中...")
    df_processed['numeric_score_margin'] = df_processed['numeric_score_margin'] * -1

    # 新しい関数でイベントIDを生成
    df_processed = assign_contextual_event_ids(df_processed)

    print("特徴量エンジニアリング完了")

"""## 6: モデル用データ準備 (フィルタリング)
フィルタリングを行う。none値を落とす等。
model_dfにフィルター後のデータフレームが格納される。
"""
if df_processed is not None:
    print("\n--- モデル用データ準備 ---")
    initial_rows = len(df_processed)

    model_df = df_processed.dropna(subset=['home_win', 'seconds_elapsed', 'numeric_score_margin', 'period', 'composite_event_id']).copy()
    model_df = model_df[model_df['period'] > 0]
    model_df = model_df[model_df['eventmsgtype'] != 12]

    filtered_rows = len(model_df)
    print(f"フィルタリング前の列数: {initial_rows}")
    print(f"フィルタリング後の列数: {filtered_rows}")
    if filtered_rows == 0:
        print("フィルターで消去したデータなし")
        model_df = pd.DataFrame() # 空にして以降の処理をスキップ

"""## 7: 特徴量とターゲットの選択、訓練/テスト分割"""
if 'model_df' in locals() and not model_df.empty:
    #エンコーディング前に元の複合IDを新しい列にコピー
    model_df['original_composite_event_id'] = model_df['composite_event_id']
    print("複合IDにラベルエンコーディングを適用中...")
    le = LabelEncoder()
    model_df['composite_event_id'] = le.fit_transform(model_df['composite_event_id'])
    
    features = ['numeric_score_margin', 'seconds_elapsed', 'home_event_id', 'visitor_event_id']
    target = 'home_win'
    print(f"\n使用特徴量: {features}")

    X = model_df[features]
    y = model_df[target].astype(int)
    groups = model_df['game_id']

    print("\nテストデータと学習データに分割...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    if groups.nunique() < 2:
        print("警告：グループシャッフルに適していません。通常の分割法を使用します。")
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
    print("model_dfが空または利用不可なので特徴選択、分割を飛ばします。")

"""##8LGMで学習"""
if 'X_train' in locals() and not X_train.empty and not X_test.empty:
    print("\n--- LightGBMモデルの訓練と評価 ---")
    
    # LightGBM用のデータセットを作成
    # categorical_featureにカテゴリとして扱いたい列名を指定
    categorical_features = ['home_event_id', 'visitor_event_id']
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, categorical_feature=categorical_features, free_raw_data=False)

    # LightGBMのパラメータを設定
    params = {
        'objective': 'binary',        # 二値分類
        'metric': 'auc',              # 評価指標としてAUCを使用 (loglossなども可)
        'boosting_type': 'gbdt',      # 標準的な勾配ブースティング
        'n_estimators': 10000,        # 木の最大数 (早期終了で最適化)
        'learning_rate': 0.02,        # 学習率
        'num_leaves': 31,             # 葉の最大数
        'max_depth': -1,              # 木の深さ (-1は制限なし)
        'seed': 42,                   # 乱数シード
        'n_jobs': -1,                 # 使用するCPUコア数 (-1は全て)
        'verbose': -1,                # ログの出力を抑制
        'colsample_bytree': 0.8,      # 木を構築する際に使用する特徴量の割合
        'subsample': 0.8,             # 木を構築する際に使用するデータの割合
    }

    # モデルの訓練
    print("モデルの訓練を開始...")
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[lgb.early_stopping(100, verbose=True), # 100ラウンド改善がなければ早期終了
                   lgb.log_evaluation(100)]              # 100ラウンドごとにログを出力
    )
    print("モデルの訓練が完了しました。")

    # テストデータで予測
    # 最も性能が良かったイテレーションで予測
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    print("予測確率を P(home_loss) から P(home_win) = 1 - P(home_loss) に変換します。")
    y_pred_proba = 1 - y_pred_proba
    y_pred_class = (y_pred_proba > 0.5).astype(int)

    # モデルの評価
    accuracy = accuracy_score(y_test, y_pred_class)
    auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    print("\n--- LightGBMモデル評価結果 ---")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  ROC AUC:     {auc:.4f}")
    print(f"  Log Loss:    {logloss:.4f}")
    print(f"  Brier Score: {brier:.4f}")

    print("\n--- 分類レポート ---")
    print(classification_report(y_test, y_pred_class))

    # 予測結果をテストデータのDataFrameに追加
    if 'model_df_test' in locals() and not model_df_test.empty:
        model_df_test['win_probability_pred'] = y_pred_proba
        print("\n--- 予測結果を追加したテストデータ (先頭5行) ---")
        display_cols = ['game_id', 'eventnum', 'numeric_score_margin', 'seconds_elapsed', 'home_win', 'win_probability_pred']
        print(model_df_test[display_cols].head())

else:
    print("\n訓練データまたはテストデータが空のため、LightGBMの訓練をスキップします。")

"""##9イベント別モメンタムの集計と可視化"""
# --- 修正後のセクション9 ---

if 'model_df_test' in locals() and 'win_probability_pred' in model_df_test.columns:
    print("\n--- イベント別モメンタム分析 (修正版) ---")

    results_df = model_df_test.sort_values(by=['game_id', 'eventnum']).copy()
    results_df['prev_win_prob'] = results_df.groupby('game_id')['win_probability_pred'].shift(1)
    results_df['win_prob_change'] = results_df['win_probability_pred'] - results_df['prev_win_prob']
    momentum_df = results_df.dropna(subset=['win_prob_change'])

    # --- 1. ホームチームのイベントが勝率に与える影響 ---
    home_events = momentum_df[momentum_df['home_event_id'] != 0].copy()
    home_momentum = home_events.groupby('home_event_id')['win_prob_change'].agg(['mean', 'count']).reset_index()
    home_momentum.rename(columns={'mean': 'avg_win_prob_change', 'count': 'event_count', 'home_event_id': 'event_id'}, inplace=True)
    home_momentum['event_name'] = home_momentum['event_id'].map(event_id_to_name_map).fillna('Unknown Event')
    home_momentum_sorted = home_momentum.sort_values(by='avg_win_prob_change', ascending=False)

    print("\n--- 【ホームチームのイベント】が勝率に与える影響ランキング ---")
    display_cols = ['event_name', 'avg_win_prob_change', 'event_count', 'event_id']
    print(home_momentum_sorted[display_cols].to_string())


    # --- 2. アウェイチームのイベントがホームチームの勝率に与える影響 ---
    visitor_events = momentum_df[momentum_df['visitor_event_id'] != 0].copy()
    visitor_momentum = visitor_events.groupby('visitor_event_id')['win_prob_change'].agg(['mean', 'count']).reset_index()
    visitor_momentum.rename(columns={'mean': 'avg_win_prob_change', 'count': 'event_count', 'visitor_event_id': 'event_id'}, inplace=True)
    visitor_momentum['event_name'] = visitor_momentum['event_id'].map(event_id_to_name_map).fillna('Unknown Event')
    visitor_momentum_sorted = visitor_momentum.sort_values(by='avg_win_prob_change', ascending=False)

    print("\n--- 【アウェイチームのイベント】がホームチームの勝率に与える影響ランキング ---")
    print(visitor_momentum_sorted[display_cols].to_string())