# data_loader.py
import sqlite3
import pandas as pd
from typing import Optional
from tqdm import tqdm

def load_and_process_pbp(db_path: str, limit_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    SQLiteデータベースからプレイバイプレイデータを読み込み、データ型を処理し、
    ホームチームの勝敗情報を計算して元のデータに結合します。
    """
    # (元のコードのセクション2にある load_and_process_pbp 関数の内容をそのままここにコピー)
    # ...
    # 以下、関数の内容
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