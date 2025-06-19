import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from config import EVENT_CATEGORIES

def train_and_interpret_model(game_features_df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    ロジスティック回帰モデルを訓練し、評価し、
    係数を「モメンタムスコア」として抽出・表示する。
    """
    print("--- モデルの訓練とスコアの算出 ---")

    # 1. 特徴量(X)とターゲット(y)を定義
    X = game_features_df[feature_cols]
    y = game_features_df[target_col]

    # 2. データを訓練用とテスト用に分割
    # stratify=y は、訓練/テストデータでの勝敗の比率を元データと同じにするためのオプション
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"訓練データ: {len(X_train)}試合, テストデータ: {len(X_test)}試合")

    # 3. ロジスティック回帰モデルを初期化し、訓練
    # C=1.0 は正則化の強さ。値を小さくするとモデルがよりシンプルになる
    # max_iterを増やすと、収束しやすくなる
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

    # 5. 係数（モメンタムスコア）を抽出し、表示
    # model.coef_[0] に各特徴量の係数が格納されている
    # 5. ホームとアウェイ別の係数（モメンタムスコア）を抽出し、表示
    
    # まず、特徴量名と係数を紐づける
    coef_map = dict(zip(feature_cols, model.coef_[0]))
    
    results = []
    for category in EVENT_CATEGORIES:
        home_feature = f'home_{category}'
        away_feature = f'away_{category}'
        #モデルが内部的にホームチームの負けを基準に係数を計算しているので、スコアを反転
        home_score = -1 * coef_map.get(home_feature, 0)
        away_score = -1 * coef_map.get(away_feature, 0)
        
        results.append({
            'Event Type': category,
            'Home Score': home_score,
            'Away Score': away_score
        })

    scores_df = pd.DataFrame(results)
    
    # ホームスコアの降順で並び替え
    scores_df = scores_df.sort_values(by='Home Score', ascending=False)
    
    print("\n--- 算出されたモメンタムスコア (ホーム/アウェイ別) ---")
    print("※Away Scoreは、アウェイチームの視点での価値に変換（係数の符号を反転）した値です。")
    print(scores_df.to_string(index=False))
    
    return model, scores_df