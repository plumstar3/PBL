import sqlite3
import pandas as pd

def load_raw_data(db_path: str, limit_rows: int = None) -> pd.DataFrame:
    """データベースからプレイ・バイ・プレイデータを読み込む"""
    print("--- データの読み込み開始 ---")
    
    try:
        conn = sqlite3.connect(db_path)
        
        query = "SELECT * FROM play_by_play"
        if limit_rows:
            query += f" LIMIT {limit_rows}"
        
        print("データを読み込み中...")
        df = pd.read_sql_query(query, conn)

        conn.close()
        
        df['homedescription'] = df['homedescription'].fillna('').astype(str)
        df['visitordescription'] = df['visitordescription'].fillna('').astype(str)
        print(f"\n{len(df)}行のデータを読み込みました。")
        return df

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        if 'conn' in locals() and conn:
            conn.close()
        return None

def get_game_outcomes(df: pd.DataFrame) -> pd.Series:
    """各試合の勝敗結果（home_win）を取得する"""
    end_game_events = df[(df['eventmsgtype'] == 13) & (df['score'].notna())].copy()
    end_game_events = end_game_events.dropna(subset=['period'])
    final_events = end_game_events.sort_values('period').groupby('game_id').last()
    
    scores_split = final_events['score'].str.split(' - ', expand=True)
    scores_split.columns = ['home_score', 'visitor_score']
    scores_split['home_score'] = pd.to_numeric(scores_split['home_score'])
    scores_split['visitor_score'] = pd.to_numeric(scores_split['visitor_score'])
    
    game_outcomes = (scores_split['home_score'] > scores_split['visitor_score']).astype(int)
    game_outcomes.name = 'home_win'
    return game_outcomes

def categorize_play(row: pd.Series) -> str | None:
    """
    プレイ・バイ・プレイの各行を、指定された7つのカテゴリに分類する。
    戻り値は 'team_eventtype' (例: 'home_2pt_success') の形式。
    """
    event_type = row['eventmsgtype']
    home_desc = row['homedescription']
    away_desc = row['visitordescription']

    if event_type == 1:
        if "3PT" in home_desc or "3PT" in away_desc:
            return 'home_3pt_success' if home_desc else 'away_3pt_success'
        else:
            return 'home_2pt_success' if home_desc else 'away_2pt_success'
    elif event_type == 2:
        return 'home_shot_miss' if home_desc else 'away_shot_miss'
    elif event_type == 3:
        if "MISS" not in home_desc and "MISS" not in away_desc:
            return 'home_ft_success' if home_desc else 'away_ft_success'
        return None
    elif event_type == 4:
        return 'home_rebound' if "REBOUND" in home_desc else 'away_rebound'
    elif event_type == 5:
        if "STEAL" in home_desc:
            return 'home_steal'
        elif "STEAL" in away_desc:
            return 'away_steal'
        else:
            if "Turnover" in home_desc:
                return 'home_unforced_turnover'
            elif "Turnover" in away_desc:
                return 'away_unforced_turnover'
    return None

def create_game_level_features(df: pd.DataFrame, event_categories: list) -> pd.DataFrame:
    """
    プレイ・バイ・プレイデータから、試合ごとの集計特徴量データフレームを作成する。
    """
    print("\n--- 試合ごとの特徴量作成開始 ---")
    
    # 1. 各プレーをカテゴリ分類
    print("各プレーをカテゴリに分類中...")
    # --- ここから修正 ---
    # tqdm.pandas() と .progress_apply() を通常の .apply() に戻す
    df['event_category'] = df.apply(categorize_play, axis=1)
    # --- 修正ここまで ---

    df_categorized = df.dropna(subset=['event_category'])
    
    # 2. 試合ごと、カテゴリごとにイベント回数を集計
    print("試合ごと、カテゴリごとにイベント回数を集計中...")
    game_event_counts = df_categorized.groupby(['game_id', 'event_category']).size().unstack(fill_value=0)

    # 3. ホームとアウェイのイベント回数を特徴量として準備
    print("ホームとアウェイのイベント回数を特徴量として準備中...")
    all_feature_columns = []
    for category in event_categories:
        all_feature_columns.append(f'home_{category}')
        all_feature_columns.append(f'away_{category}')
        
    for col in all_feature_columns:
        if col not in game_event_counts:
            game_event_counts[col] = 0
            
    feature_df = game_event_counts[all_feature_columns]

    # 4. 試合の勝敗結果を結合
    print("試合の勝敗結果を結合中...")
    game_outcomes = get_game_outcomes(df)
    final_df = feature_df.join(game_outcomes).dropna()
    final_df['home_win'] = final_df['home_win'].astype(int)

    print(f"--- 特徴量作成完了 ---")
    print(f"最終的なデータセットの形状: {final_df.shape}")
    
    return final_df