from config import DB_PATH, LIMIT_ROWS, EVENT_CATEGORIES, MODEL_FEATURES
from data_processor import load_raw_data, create_game_level_features
from model import train_and_interpret_model
from game_analyzer import export_game_analysis_to_excel

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
    
    # 3. モデルを訓練し、モメンタムスコアを算出する処理を追加
    if not game_features_df.empty:
        model, scores_df = train_and_interpret_model(
            game_features_df, 
            feature_cols=MODEL_FEATURES, 
            target_col='home_win'
        )


if __name__ == "__main__":
    main()