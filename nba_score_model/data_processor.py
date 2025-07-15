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
    
    # 列名をデータの表記（away - home）に合わせて正しく設定
    scores_split.columns = ['away_score', 'home_score'] 
    
    scores_split['home_score'] = pd.to_numeric(scores_split['home_score'])
    scores_split['away_score'] = pd.to_numeric(scores_split['away_score'])
    
    game_outcomes = (scores_split['home_score'] > scores_split['away_score']).astype(int)
    game_outcomes.name = 'home_win'
    return game_outcomes

def categorize_play(row: pd.Series) -> list:
    """
    1つのプレーから、関連する全てのイベントカテゴリをリストとして返す。
    """
    categories = []
    event_type = row['eventmsgtype']
    home_desc = row['homedescription']
    away_desc = row['visitordescription']
    
    # アクションを起こしたのがホームかアウェイかを判定
    actor = 'home' if home_desc else 'away'
    opponent = 'away' if actor == 'home' else 'home'
    actor_desc = home_desc if actor == 'home' else away_desc
    
    # --- シュート成功 (eventmsgtype: 1) ---
    if event_type == 1:
        if "3PT" in actor_desc:
            categories.append(f'{actor}_3pt_success')
        else:
            categories.append(f'{actor}_2pt_success')

    # --- シュート失敗 (eventmsgtype: 2) ---
    elif event_type == 2:
        # ブロックを優先して判定
        if "BLOCK" in home_desc or "BLOCK" in away_desc:
            blocker = 'home' if "BLOCK" in home_desc else 'away'
            categories.append(f'{blocker}_block')
        else:
            # ブロックされなかった純粋なシュートミス
            categories.append(f'{actor}_shot_miss_unblocked')

    # --- フリースロー (eventmsgtype: 3) ---
    elif event_type == 3:
        if "MISS" in actor_desc:
            categories.append(f'{actor}_ft_miss')
        else:
            categories.append(f'{actor}_ft_success')

    # --- リバウンド (eventmsgtype: 4) ---
    elif event_type == 4:
        rebounder = 'home' if home_desc else 'away'
        categories.append(f'{rebounder}_rebound')

        
    # --- ターンオーバー (eventmsgtype: 5) ---
    elif event_type == 5:
        # スティールを優先して判定
        if "STEAL" in home_desc or "STEAL" in away_desc:
            stealer = 'home' if "STEAL" in home_desc else 'away'
            categories.append(f'{stealer}_steal')
        else:
            # スティール以外のターンオーバー
            turnover_committer = 'home' if "Turnover" in home_desc else 'away'
            categories.append(f'{turnover_committer}_unforced_turnover')
            
    # --- ファウル (eventmsgtype: 6) ---
    elif event_type == 6:
        # 今回は単純にファウルとしてカウント
        fouler = 'home' if "FOUL" in home_desc else 'away'
        categories.append(f'{fouler}_foul')

    return categories

def create_game_level_features(df: pd.DataFrame, event_categories: list) -> pd.DataFrame:
    """
    プレイ・バイ・プレイデータから、試合ごとの集計特徴量データフレームを作成する。
    """
    print("\n--- 試合ごとの特徴量作成開始 ---")
    
    # 1. 各プレーをカテゴリ分類（複数のカテゴリが返される可能性がある）
    print("各プレーをカテゴリに分類中...")
    df['event_categories'] = df.apply(categorize_play, axis=1)

    # 2. カテゴリリストを個別の行に展開（explode）
    # 例: ['home_2pt_success', 'home_assist'] というリストが2行のデータになる
    print("イベントデータを展開中...")
    df_exploded = df.explode('event_categories')
    df_categorized = df_exploded.dropna(subset=['event_categories'])

    # 3. 試合ごと、カテゴリごとにイベント回数を集計
    print("試合ごと、カテゴリごとにイベント回数を集計中...")
    game_event_counts = df_categorized.groupby(['game_id', 'event_categories']).size().unstack(fill_value=0)

    # 4. ホームとアウェイのイベント回数を特徴量として準備
    print("ホームとアウェイのイベント回数を特徴量として準備中...")
    all_feature_columns = []
    for category in event_categories:
        all_feature_columns.append(f'home_{category}')
        all_feature_columns.append(f'away_{category}')
        
    for col in all_feature_columns:
        if col not in game_event_counts:
            game_event_counts[col] = 0
            
    feature_df = game_event_counts.reindex(columns=all_feature_columns, fill_value=0)

    # 5. 試合の勝敗結果を結合
    print("試合の勝敗結果を結合中...")
    game_outcomes = get_game_outcomes(df)
    final_df = feature_df.join(game_outcomes).dropna()
    final_df['home_win'] = final_df['home_win'].astype(int)

    print(f"--- 特徴量作成完了 ---")
    print(f"最終的なデータセットの形状: {final_df.shape}")
    
    return final_df