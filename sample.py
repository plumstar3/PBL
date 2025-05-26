import sqlite3
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss, classification_report
import numpy as np
from typing import Optional, Tuple, List
import os
from tqdm import tqdm

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
# db_file = r'C:\Users\amilu\Projects\vsCodeFile\PBL\nba.sqlite' # 研究室PC用path
db_file = r'C:\Programing\PBL\nba.sqlite' # ノートPC用path

# 処理する行数を設定 (Noneにすると全データを対象)
# limit_rows = 500000 # 動作確認には50万行程度がおすすめ
limit_rows = 45616 # 元のコードの値
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

print(df_processed['composite_event_id'])