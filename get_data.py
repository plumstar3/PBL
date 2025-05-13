import sqlite3
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss, classification_report
import numpy as np
from typing import Optional

def load_and_process_pbp(db_path: str, limit_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    SQLiteデータベースからプレイバイプレイデータを読み込み、データ型を処理し、
    ホームチームの勝敗情報を計算して元のデータに結合します。

    Args:
        db_path (str): nba.sqliteファイルへのパス。
        limit_rows (Optional[int]): 読み込む最大行数。Noneの場合は全行読み込みます。
                                     デフォルトは None。

    Returns:
        Optional[pd.DataFrame]: 処理済みのプレイバイプレイデータ（home_winカラム付き）。
                                エラー発生時は None を返します。
                                limit_rowsが指定された場合、home_winカラムは
                                読み込んだデータ範囲内の試合しか反映されない可能性があります。
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

        # 2. SQLクエリの構築
        query = """
        SELECT
            game_id, eventnum, eventmsgtype, period, pctimestring,
            homedescription, neutraldescription, visitordescription,
            score, scoremargin
        FROM
            play_by_play
        """
        if limit_rows:
            query += f" LIMIT {limit_rows};"
        else:
            query += ";"

        print("Executing SQL query...")
        # 3. クエリ実行とDataFrameへの読み込み
        df = pd.read_sql_query(query, conn)
        print(f"Successfully loaded {len(df)} rows into DataFrame.")

        # 4. データベース接続を閉じる
        conn.close()
        print("Database connection closed.")

        # 5. データ型の確認と変換
        print("Processing data types...")
        df['game_id'] = df['game_id'].astype(str)
        df['pctimestring'] = df['pctimestring'].astype(str)
        df['score'] = df['score'].astype(str)
        df['scoremargin'] = df['scoremargin'].astype(str)
        df['eventnum'] = pd.to_numeric(df['eventnum'], errors='coerce')
        df['eventmsgtype'] = pd.to_numeric(df['eventmsgtype'], errors='coerce')
        df['period'] = pd.to_numeric(df['period'], errors='coerce')
        desc_cols = ['homedescription', 'neutraldescription', 'visitordescription']
        for col in desc_cols:
            df[col] = df[col].fillna('')
        print("Data type processing complete.")

        # 6. 読み込んだデータの情報表示 (オプション)
        # print("\nDataFrame Info after loading and type conversion:")
        # df.info()
        # print("\nFirst 5 rows:")
        # print(df.head())

        # --- 最終スコアの取得と勝敗判定 ---
        print("\nAttempting to determine game outcomes...")
        game_outcomes = pd.Series(dtype=int) # 空のSeriesを初期化

        # 7. 試合終了イベントのフィルタリング
        end_game_events = df[(df['eventmsgtype'] == 13) & (df['score'].str.contains(' - ', na=False))].copy()

        if not end_game_events.empty:
            # 8. 各ゲームの最終ピリオドの終了イベントを取得
            end_game_events = end_game_events.dropna(subset=['period'])
            if not end_game_events.empty:
                end_game_events['period'] = end_game_events['period'].astype(int)
                final_events = end_game_events.sort_values('period').groupby('game_id').last()

                # 9. スコアを分割して数値に変換
                if 'score' in final_events.columns:
                    scores_split = final_events['score'].str.split(' - ', expand=True)
                    scores_split.columns = ['home_score', 'visitor_score']
                    scores_split['home_score'] = pd.to_numeric(scores_split['home_score'], errors='coerce')
                    scores_split['visitor_score'] = pd.to_numeric(scores_split['visitor_score'], errors='coerce')
                    scores_split = scores_split.dropna(subset=['home_score', 'visitor_score'])

                    # 10. ホームチームの勝敗を判定
                    if not scores_split.empty:
                        scores_split['home_win'] = (scores_split['home_score'] > scores_split['visitor_score']).astype(int)
                        game_outcomes = scores_split['home_win']
                        print(f"Determined outcomes for {len(game_outcomes)} games.")
                        if limit_rows:
                            print("[Warning] Game outcomes may be incomplete due to the row limit.")
                    else:
                        print("Could not determine winners after score processing.")
                else:
                    print("No valid numeric scores found after splitting.")
            else:
                 print("No end game events with valid 'period' found.")
        else:
             print("No end game events (eventmsgtype=13) with valid scores found.")

        # 11. 元のDataFrameに結合
        print("Merging game outcomes back to the main DataFrame...")
        if not game_outcomes.empty:
             df_with_outcome = df.merge(game_outcomes.rename('home_win'), on='game_id', how='left')
             # 勝敗が判定できなかったゲームの home_win は NaN になる
             # 必要に応じて NaN を処理 (例: 0 や -1 で埋めるなど)
             # df_with_outcome['home_win'] = df_with_outcome['home_win'].fillna(-1) # 例: 不明な試合は-1
             print("Merge complete.")
        else:
             print("No game outcomes determined, adding 'home_win' column with NaN.")
             df['home_win'] = pd.NA # 全て NaN の列を追加
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
        print("Please ensure the file exists and the path is correct.")
        return None
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error details: {e}")
        return None

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
        return None # 無効なピリオド

    seconds_in_period = parse_time_to_seconds(pctimestring)
    if seconds_in_period is None:
        return None # 時間文字列が無効

    # ピリオドごとの秒数 (Regulation: 12min, OT: 5min)
    seconds_per_period = 720 if period <= 4 else 300
    seconds_elapsed_in_current_period = seconds_per_period - seconds_in_period

    # エラーチェック: 経過時間がマイナスまたはピリオド時間より大きくならないか
    if seconds_elapsed_in_current_period < 0 or seconds_elapsed_in_current_period > seconds_per_period:
         # print(f"Warning: Invalid elapsed time calculated for {row['game_id']} P{period} {pctimestring}")
         return None # 無効な計算結果

    # 経過総秒数
    if period <= 4:
        total_seconds_elapsed = (period - 1) * 720 + seconds_elapsed_in_current_period
    else: # Overtime
        total_seconds_elapsed = 4 * 720 + (period - 5) * 300 + seconds_elapsed_in_current_period

    return total_seconds_elapsed

def process_score_margin(margin_str):
    """ scoremargin を数値に変換 ('TIE' -> 0) """
    if margin_str == 'TIE':
        return 0
    elif pd.isna(margin_str) or margin_str == '':
         return None # 欠損値はNoneのまま
    else:
        try:
            # プラス記号を削除してから数値に変換
            return int(str(margin_str).replace('+', ''))
        except ValueError:
            return None # 数値に変換できない場合はNone

# --- 関数の使用例 ---
if __name__ == "__main__":
    # 例1: 最初の 50000 行を読み込む
    #ノートPC用path
    #db_file = 'C:\Workspace\PBL\\nba.sqlite'

    #研究室PC用path
    db_file = r'C:\Users\amilu\Projects\vsCodeFile\PBL\nba.sqlite'

    limit_rows = 45616
    # 1. Load Data
    df_processed = load_and_process_pbp(db_file, limit_rows=limit_rows)

    if df_processed is not None:
        print("\n--- Feature Engineering ---")

        # 2. Feature Engineering
        # 2.1 Calculate total seconds elapsed
        print("Calculating total seconds elapsed...")
        df_processed['seconds_elapsed'] = df_processed.apply(calculate_seconds_elapsed, axis=1)

        # 2.2 Process score margin
        print("Processing score margin...")
        df_processed['numeric_score_margin'] = df_processed['scoremargin'].apply(process_score_margin)

        # 3. Data Preparation for Modeling
        print("\n--- Data Preparation for Modeling ---")

        # 3.1 Filter out invalid rows
        # - home_winがNaN (勝敗不明の試合)
        # - seconds_elapsedがNaN (時間計算エラー)
        # - numeric_score_marginがNaN (スコアマージン処理エラー or 元々Null)
        # - periodがNaNまたは0以下
        model_df = df_processed.dropna(subset=['home_win', 'seconds_elapsed', 'numeric_score_margin', 'period'])
        model_df = model_df[model_df['period'] > 0] # periodが1以上のプレイのみ

        # eventmsgtype=12 (Start Period) のデータは予測に意味がないので除外 (その時点での状態がないため)
        model_df = model_df[model_df['eventmsgtype'] != 12]

        print(f"Filtered data for modeling: {len(model_df)} rows")

        if len(model_df) > 0:
            # 3.2 Select features and target
            features = ['numeric_score_margin', 'seconds_elapsed']
            target = 'home_win'
            X = model_df[features]
            y = model_df[target].astype(int) # ターゲットを整数型に

            # 3.3 Split data using GroupShuffleSplit to keep games together
            # test_size=0.2 means 20% of the *groups* (games) go to the test set
            print("Splitting data into training and testing sets (game-aware)...")
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(X, y, groups=model_df['game_id']))

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            print(f"Training set size: {len(X_train)}")
            print(f"Testing set size: {len(X_test)}")

            # 4. Model Training
            print("\n--- Model Training ---")

            # 4.1 Scale features
            print("Scaling features...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test) # Use transform only on test data

            # 4.2 Train Logistic Regression model
            print("Training Logistic Regression model...")
            model = LogisticRegression(random_state=42)
            model.fit(X_train_scaled, y_train)
            print("Model training complete.")

            # 5. Prediction and Evaluation
            print("\n--- Prediction and Evaluation ---")

            # 5.1 Predict probabilities on the test set
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Probability of home_win=1
            y_pred_class = model.predict(X_test_scaled) # Class prediction (0 or 1)

            # 5.2 Evaluate the model
            accuracy = accuracy_score(y_test, y_pred_class)
            auc = roc_auc_score(y_test, y_pred_proba)
            logloss = log_loss(y_test, y_pred_proba)
            brier = brier_score_loss(y_test, y_pred_proba)

            print(f"Model Evaluation Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  ROC AUC:  {auc:.4f}")
            print(f"  Log Loss: {logloss:.4f}")
            print(f"  Brier Score: {brier:.4f}")

            print("\nClassification Report:")
            print(classification_report(y_test, y_pred_class))

            # --- Example: Add predictions back to the test part of the dataframe ---
            # This is useful for the next step (analyzing probability changes)
            model_df_test = model_df.iloc[test_idx].copy()
            model_df_test['win_probability_pred'] = y_pred_proba

        
            if 'numeric_score_margin' in model_df_test.columns:
                print("\nReversing the sign of 'numeric_score_margin' in model_df_test...")
                # 各要素に -1 を掛けて正負を反転
                # この操作は、numeric_score_margin がホームチーム視点かビジターチーム視点かを
                # 統一したり、逆転させたりする場合に行います。
                model_df_test['numeric_score_margin_invation'] = model_df_test['numeric_score_margin'] * -1
                print("Sign reversal of 'numeric_score_margin_invation' complete.")

                # 変更後の確認 (任意)
                print("\nFirst 5 rows of test data after reversing 'numeric_score_margin_invation' sign:")
                # 元の scoremargin と比較
                print(model_df_test[['game_id', 'eventnum', 'scoremargin', 'numeric_score_margin', 'win_probability_pred', 'numeric_score_margin_invation']].head())
            else:
                print("\nWarning: 'numeric_score_margin' column not found in model_df_test. Cannot reverse sign.")

            print("\nFirst 5 rows of test data with predicted win probability:")
            print(model_df_test[['game_id', 'eventnum', 'period', 'pctimestring', 'score', 'scoremargin', 'home_win', 'win_probability_pred', 'numeric_score_margin']].head())

             # --- CSV出力部分 ---
            # check if model_df_test exists and is a DataFrame
            if 'model_df_test' in locals() and isinstance(model_df_test, pd.DataFrame) and not model_df_test.empty:
                output_csv_path = 'win_probability_predictions.csv'
                print(f"\n--- Exporting Test Predictions to CSV ---")
                try:
                    # Ensure the directory exists if specifying a full path
                    # os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
                    model_df_test.to_csv(output_csv_path, index=False, encoding='utf-8')
                    print(f"Successfully exported test predictions to: {output_csv_path}")
                except Exception as e:
                    print(f"Error exporting to CSV: {e}")
            else:
                print("\nSkipping CSV export because 'model_df_test' was not generated, is empty, or is not a DataFrame.")

        else:
            print("Not enough valid data remaining after filtering to train a model.")

    else:
        print("Failed to load data. Cannot proceed with modeling.")

    print("\n--- Win probability prediction script finished ---")