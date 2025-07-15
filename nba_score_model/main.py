from config import DB_PATH, LIMIT_ROWS, EVENT_CATEGORIES, MODEL_FEATURES
from data_processor import load_raw_data, create_game_level_features
from model import train_and_interpret_model
from game_analyzer import create_quarterly_data, analyze_comeback_cases, run_gee_model

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
        model, scores_df, test_game_ids = train_and_interpret_model(
            game_features_df, 
            feature_cols=MODEL_FEATURES, 
            target_col='home_win'
        )
        # 4. 【今回追加】Excelファイルへの出力処理
        #    テストデータセットの最初の試合をサンプルとして出力する
        # if test_game_ids: # テストデータが存在する場合のみ実行
        #     sample_game_id = '0029600458'
        #     export_game_analysis_to_excel(
        #         game_id=sample_game_id,
        #         raw_df=df_raw,      # 生のPBPデータ
        #         scores_df=scores_df   # 訓練済みモデルから得たスコア
        #     )

        # ステップ4: クォーターごとの分析用データを作成
        quarterly_df = create_quarterly_data(
            test_game_ids=test_game_ids,
            raw_df=df_raw,
            scores_df=scores_df
        )

        if not quarterly_df.empty:
            # ステップ5: 逆転可能性の比較分析を実行
            analyze_comeback_cases(quarterly_df)

            # ステップ6: 混合効果モデル分析を実行
            run_gee_model(quarterly_df)


if __name__ == "__main__":
    main()