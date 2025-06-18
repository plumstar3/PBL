from config import DB_PATH, LIMIT_ROWS, EVENT_CATEGORIES
from data_processor import load_raw_data, create_game_level_features

def main():
    """
    プロジェクトのメイン実行関数
    """
    # 1. 生データをデータベースから読み込み
    df_raw = load_raw_data(DB_PATH, limit_rows=LIMIT_ROWS)
    
    # 2. 試合ごとの特徴量データを作成
    game_features_df = create_game_level_features(df_raw, event_categories=EVENT_CATEGORIES)
    
    print("\n--- 前処理完了 ---")
    print("作成された特徴量データ（先頭5行）:")
    print(game_features_df.head())
    
    # この後、この game_features_df を使ってロジスティック回帰モデルを訓練する
    # (次回のステップで実装)

if __name__ == "__main__":
    main()