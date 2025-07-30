# data_processor.py

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
    
    scores_split.columns = ['home_score', 'away_score'] 
    
    scores_split['home_score'] = pd.to_numeric(scores_split['home_score'])
    scores_split['away_score'] = pd.to_numeric(scores_split['away_score'])
    
    game_outcomes = (scores_split['home_score'] > scores_split['away_score']).astype(int)
    game_outcomes.name = 'home_win'
    return game_outcomes

def categorize_play(row: pd.Series) -> list:
    """
    1つのプレーから、クォーター情報を付加したカテゴリをリストとして返す。
    """
    categories = []
    event_type = row['eventmsgtype']
    period = row['period']
    home_desc = row['homedescription']
    away_desc = row['visitordescription']

    # --- ▼▼▼ ここからが重要な修正箇所 ▼▼▼ ---
    if pd.isna(period) or period < 1:
        return categories
    
    # 延長戦（OT）は第4クォーターとして扱う
    quarter = int(min(period, 4))
    # --- ▲▲▲ 修正ここまで ▲▲▲ ---

    actor = 'home' if home_desc else 'away'
    actor_desc = home_desc if actor == 'home' else away_desc
    
    # 各カテゴリ名に {quarter}q_ というプレフィックスを付ける
    if event_type == 1:
        if "3PT" in actor_desc.upper():
            categories.append(f'{actor}_{quarter}q_3pt_success')
        else:
            categories.append(f'{actor}_{quarter}q_2pt_success')
        if "ASSIST" in actor_desc.upper():
            categories.append(f'{actor}_{quarter}q_assist')

    elif event_type == 2:
        if "BLOCK" in home_desc.upper() or "BLOCK" in away_desc.upper():
            blocker = 'home' if "BLOCK" in home_desc.upper() else 'away'
            categories.append(f'{blocker}_{quarter}q_block')
        else:
            categories.append(f'{actor}_{quarter}q_shot_miss_unblocked')

    elif event_type == 3:
        if "MISS" in actor_desc.upper():
            categories.append(f'{actor}_{quarter}q_ft_miss')
        else:
            categories.append(f'{actor}_{quarter}q_ft_success')

    elif event_type == 4:
        rebounder = 'home' if home_desc else 'away'
        categories.append(f'{rebounder}_{quarter}q_rebound')
        
    elif event_type == 5:
        if "STEAL" in home_desc.upper() or "STEAL" in away_desc.upper():
            stealer = 'home' if "STEAL" in home_desc.upper() else 'away'
            categories.append(f'{stealer}_{quarter}q_steal')
        else:
            turnover_committer = 'home' if "TURNOVER" in home_desc.upper() else 'away'
            categories.append(f'{turnover_committer}_{quarter}q_unforced_turnover')
            
    elif event_type == 6:
        fouler = 'home' if "FOUL" in home_desc.upper() else 'away'
        categories.append(f'{fouler}_{quarter}q_foul')

    return categories

def create_game_level_features(df: pd.DataFrame, event_categories: list) -> pd.DataFrame:
    """
    プレイ・バイ・プレイデータから、試合ごとの集計特徴量データフレームを作成する。
    """
    print("\n--- 試合ごとの特徴量作成開始 ---")
    
    print("各プレーをカテゴリに分類中...")
    df['event_categories'] = df.apply(categorize_play, axis=1)

    print("イベントデータを展開中...")
    df_exploded = df.explode('event_categories')
    df_categorized = df_exploded.dropna(subset=['event_categories'])

    print("試合ごと、カテゴリごとにイベント回数を集計中...")
    game_event_counts = df_categorized.groupby(['game_id', 'event_categories']).size().unstack(fill_value=0)

    print("ホームとアウェイのイベント回数を特徴量として準備中...")
    
    # config.pyで生成される詳細な列名リストをここで作成
    all_feature_columns = []
    for category in event_categories:
        for q in [1, 2, 3, 4]:
            all_feature_columns.append(f'home_{q}q_{category}')
            all_feature_columns.append(f'away_{q}q_{category}')

    # 存在しない列を0で埋めるためにreindexを使用
    feature_df = game_event_counts.reindex(columns=all_feature_columns, fill_value=0)

    print("試合の勝敗結果を結合中...")
    game_outcomes = get_game_outcomes(df)
    final_df = feature_df.join(game_outcomes).dropna()
    final_df['home_win'] = final_df['home_win'].astype(int)

    print(f"--- 特徴量作成完了 ---")
    print(f"最終的なデータセットの形状: {final_df.shape}")
    
    return final_df