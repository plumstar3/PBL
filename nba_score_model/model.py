# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from config import EVENT_CATEGORIES # configからEVENT_CATEGORIESをインポート

def train_and_interpret_model(game_features_df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    ロジスティック回帰モデルを訓練し、評価し、
    係数を「モメンタムスコア」として抽出・表示する。
    """
    print("--- モデルの訓練とスコアの算出 ---")

    # ステップ1, 2, 3は変更なし
    X = game_features_df[feature_cols]
    y = game_features_df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"訓練データ: {len(X_train)}試合, テストデータ: {len(X_test)}試合")
    model = LogisticRegression(random_state=42, C=1.0, max_iter=1000)
    print("モデルの訓練を開始...")
    model.fit(X_train, y_train)
    print("モデルの訓練が完了しました。")

    # 4. テストデータでモデルの性能を評価
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n--- モデル性能評価 ---")
    print(f"予測正解率 (Accuracy): {accuracy:.4f}")
    print("\n分類レポート:")
    print(classification_report(y_test, y_pred))

    # --- ▼▼▼ ここからが今回の修正箇所 ▼▼▼ ---
    # 5. クォーター別の係数（モメンタムスコア）を抽出し、表示
    
    # まず、特徴量名と係数を紐づける
    coef_map = dict(zip(feature_cols, model.coef_[0]))
    
    results = []
    # イベントカテゴリとクォーターでループ処理
    for category in EVENT_CATEGORIES:
        for q in [1, 2, 3, 4]:
            home_feature = f'home_{q}q_{category}'
            away_feature = f'away_{q}q_{category}'
            
            # 正しいクォーター別の特徴量名でスコアを取得
            home_score = coef_map.get(home_feature, 0)
            away_score = coef_map.get(away_feature, 0)
            
            results.append({
                'Event Type': category,
                'Quarter': q,
                'Home Score': home_score,
                'Away Score': away_score
            })

    scores_df = pd.DataFrame(results)
    
    # イベントタイプ、クォーターの順で並び替え
    scores_df = scores_df.sort_values(by=['Event Type', 'Quarter'])
    
    print("\n--- 算出されたモメンタムスコア (クォーター別) ---")
    print(scores_df.to_string(index=False))
    # --- ▲▲▲ 修正ここまで ▲▲▲ ---

    return model, scores_df, X_test.index.tolist()