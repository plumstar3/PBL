import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss, classification_report

def prepare_data_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """モデル投入用にデータをフィルタリングする"""
    print("\n--- モデル用データ準備 ---")
    initial_rows = len(df)
    
    model_df = df.dropna(
        subset=['home_win', 'seconds_elapsed', 'numeric_score_margin', 'period', 'composite_event_id']
    ).copy()
    model_df = model_df[model_df['period'] > 0]
    model_df = model_df[model_df['eventmsgtype'] != 12] # ピリオド開始イベントを除外

    filtered_rows = len(model_df)
    print(f"フィルタリング: {initial_rows}行 -> {filtered_rows}行")
    return model_df

def split_data(df: pd.DataFrame, features: list, target: str) -> tuple:
    """データを訓練用とテスト用に分割する"""
    print("\n--- データ分割 ---")
    X = df[features]
    y = df[target].astype(int)
    groups = df['game_id']

    # エンコーディングとは無関係な分析用のIDを保持
    df['original_composite_event_id'] = df['composite_event_id']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model_df_test = df.iloc[test_idx].copy()

    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, model_df_test

def train_and_evaluate(X_train, y_train, X_test, y_test, params, categorical_features) -> tuple:
    """LightGBMモデルの訓練、評価、予測を行う"""
    print("\n--- LightGBMモデルの訓練と評価 ---")
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, categorical_feature=categorical_features)

    print("モデルの訓練を開始...")
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(100)]
    )
    print("モデルの訓練が完了しました。")

    # --- ここから修正 ---

    # 1. モデル評価用には、model.predict() が返す「生の」予測確率を使用する
    y_pred_proba_for_eval = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_class_for_eval = (y_pred_proba_for_eval > 0.5).astype(int)

    # モデル評価（生の確率で計算）
    print("\n--- LightGBMモデル評価結果 ---")
    accuracy = accuracy_score(y_test, y_pred_class_for_eval)
    auc = roc_auc_score(y_test, y_pred_proba_for_eval) # ★生の確率を使用
    logloss = log_loss(y_test, y_pred_proba_for_eval)   # ★生の確率を使用
    brier = brier_score_loss(y_test, y_pred_proba_for_eval) # ★生の確率を使用
    
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  ROC AUC:     {auc:.4f}")
    print(f"  Log Loss:    {logloss:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print("\n--- 分類レポート ---")
    print(classification_report(y_test, y_pred_class_for_eval))

    # 2. モメンタム分析用には、「反転させた」予測確率を使用する
    y_pred_proba_for_analysis = 1 - y_pred_proba_for_eval

    # 戻り値として「分析用の確率」を main.py に渡す
    return model, y_pred_proba_for_analysis