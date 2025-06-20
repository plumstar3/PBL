import pandas as pd
from config import DB_PATH, LIMIT_ROWS, FEATURES, TARGET, LGB_PARAMS, CATEGORICAL_FEATURES, EVENT_ID_TO_NAME_MAP
from data_loader import load_and_process_pbp
from feature_engineering import create_features
from model_trainer import prepare_data_for_model, split_data, train_and_evaluate
from analysis import analyze_momentum

def main():
    """
    NBAプレイバイプレイデータからモメンタムを分析するメインパイプライン
    """
    print("--- パイプライン開始 ---")

    # 1. データの読み込み
    df_raw = load_and_process_pbp(DB_PATH, limit_rows=LIMIT_ROWS)
    if df_raw is None:
        print("データ読み込みに失敗しました。処理を終了します。")
        return

    # 2. 特徴量エンジニアリング
    df_featured = create_features(df_raw)

    # 3. モデル用データの準備（フィルタリング）
    model_df = prepare_data_for_model(df_featured)
    if model_df.empty:
        print("モデル用のデータがありません。処理を終了します。")
        return

    # 4. 訓練/テスト分割
    X_train, X_test, y_train, y_test, model_df_test = split_data(
        model_df, features=FEATURES, target=TARGET
    )

    # 5. モデルの訓練と評価
    model, y_pred_proba = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        params=LGB_PARAMS,
        categorical_features=CATEGORICAL_FEATURES
    )

    # 6. モメンタム分析
    model_df_test['win_probability_pred'] = y_pred_proba
    analyze_momentum(model_df_test, event_map=EVENT_ID_TO_NAME_MAP)

    print("\n--- パイプライン終了 ---")

if __name__ == "__main__":
    main()