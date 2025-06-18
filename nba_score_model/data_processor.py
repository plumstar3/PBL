import sqlite3
import pandas as pd
from tqdm import tqdm

def load_raw_data(db_path: str, limit_rows: int = None) -> pd.DataFrame:
    """データベースからプレイ・バイ・プレイデータを読み込む"""
    print("--- データの読み込み開始 ---")
    conn = sqlite3.connect(db_path)
    
    query = "SELECT * FROM play_by_play"
    if limit_rows:
        query += f" LIMIT {limit_rows}"
        
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # データ型の基本処理
    df['homedescription'] = df['homedescription'].fillna('').astype(str)
    df['visitordescription'] = df['visitordescription'].fillna('').astype(str)
    print(f"{len(df)}行のデータを読み込みました。")
    return df

def get_game_outcomes(df: pd.DataFrame) -> pd.Series:
    """各試合の勝敗結果（home_win）を取得する"""
    end_game_events = df[(df['eventmsgtype'] == 13) & (df['score'].notna())].copy()
    end_game_events = end_game_events.dropna(subset=['period'])
    #group_idの中で最も後のピリオドを採用
    final_events = end_game_events.sort_values('period').groupby('game_id').last()
    
    #スコアの文字列をホームアウェイの数値に分割
    scores_split = final_events['score'].str.split(' - ', expand=True)
    scores_split.columns = ['home_score', 'visitor_score']
    scores_split['home_score'] = pd.to_numeric(scores_split['home_score'])
    scores_split['visitor_score'] = pd.to_numeric(scores_split['visitor_score'])
    
    game_outcomes = (scores_split['home_score'] > scores_split['visitor_score']).astype(int)
    game_outcomes.name = 'home_win'
    return game_outcomes

def categorize_play(row: pd.Series) -> str | None:
    """
    プレイ・バイ・プレイの各行を、7つのカテゴリに分類する。
    戻り値は 'team_eventtype' (例: 'home_2pt_success') の形式。
    """
    event_type = row['eventmsgtype']
    home_desc = row['homedescription']
    away_desc = row['visitordescription']

    # --- シュート成功 (eventmsgtype: 1) ---
    if event_type == 1:
        if "3PT" in home_desc or "3PT" in away_desc:
            return 'home_3pt_success' if home_desc else 'away_3pt_success'
        else: # 3PTでない成功は全て2P成功と見なす
            return 'home_2pt_success' if home_desc else 'away_2pt_success'

    # --- シュート失敗 (eventmsgtype: 2) ---
    elif event_type == 2:
        return 'home_shot_miss' if home_desc else 'away_shot_miss'

    # --- フリースロー (eventmsgtype: 3) ---
    elif event_type == 3:
        # 失敗（MISS）が含まれていないものを成功と見なす
        if "MISS" not in home_desc and "MISS" not in away_desc:
            return 'home_ft_success' if home_desc else 'away_ft_success'
        return None # FT失敗は今回カウントしない

    # --- リバウンド (eventmsgtype: 4) ---
    elif event_type == 4:
        return 'home_rebound' if "REBOUND" in home_desc else 'away_rebound'
        
    # --- ターンオーバー (eventmsgtype: 5) ---
    elif event_type == 5:
        # スティールかどうかを先に判断する
        if "STEAL" in home_desc:
            return 'home_steal' # ホームがスティールした
        elif "STEAL" in away_desc:
            return 'away_steal' # アウェイがスティールした
        else:
            # スティールでなければ、純粋なターンオーバー
            # "Turnover"という記述がある側がミスをしたチーム
            if "Turnover" in home_desc:
                return 'home_unforced_turnover'
            elif "Turnover" in away_desc:
                return 'away_unforced_turnover'
    
    return None # 上記以外のイベントは対象外

def create_game_level_features(df: pd.DataFrame, event_categories: list) -> pd.DataFrame:
    """
    プレイ・バイ・プレイデータから、試合ごとの集計特徴量データフレームを作成する。
    """
    print("\n--- 試合ごとの特徴量作成開始 ---")
    
    # 1. 各プレーをカテゴリ分類
    print("各プレーをカテゴリに分類中...")
    tqdm.pandas(desc="Categorizing plays")
    df['event_category'] = df.progress_apply(categorize_play, axis=1)

    # 対象カテゴリのプレーのみに絞る
    df_categorized = df.dropna(subset=['event_category'])
    
    # 2. 試合ごと、カテゴリごとにイベント回数を集計
    print("試合ごと、カテゴリごとにイベント回数を集計中...")
    game_event_counts = df_categorized.groupby(['game_id', 'event_category']).size().unstack(fill_value=0)

    # 3. イベント回数の差分を計算
    print("ホームとアウェイのイベント回数の差分を計算中...")
    feature_df = pd.DataFrame(index=game_event_counts.index)

    for category in event_categories:
        home_col = f'home_{category}'
        away_col = f'away_{category}'
        
        # データに存在しないカテゴリの列を0で埋めて作成
        if home_col not in game_event_counts: game_event_counts[home_col] = 0
        if away_col not in game_event_counts: game_event_counts[away_col] = 0
            
        feature_df[f'diff_{category}'] = game_event_counts[home_col] - game_event_counts[away_col]

    # 4. 試合の勝敗結果を結合
    print("試合の勝敗結果を結合中...")
    game_outcomes = get_game_outcomes(df)
    final_df = feature_df.join(game_outcomes).dropna()
    final_df['home_win'] = final_df['home_win'].astype(int)

    print(f"--- 特徴量作成完了 ---")
    print(f"最終的なデータセットの形状: {final_df.shape}")
    
    return final_df