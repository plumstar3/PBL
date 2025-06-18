# analysis.py
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def prepare_momentum_df(df_with_preds: pd.DataFrame) -> pd.DataFrame:
    """予測結果を持つDFから、モメンタム計算の準備ができたDFを返す"""
    print("\n--- モメンタム分析用データ準備 ---")
    # 必ず試合のプレー順にソートする
    results_df = df_with_preds.sort_values(by=['game_id', 'eventnum'])
    # 1つ前のプレーの勝率を計算
    results_df['prev_win_prob'] = results_df.groupby('game_id')['win_probability_pred'].shift(1)
    # 勝率の変化を計算
    results_df['win_prob_change'] = results_df['win_probability_pred'] - results_df['prev_win_prob']
    # 最初のプレー（変化を計算できない）を除外
    momentum_df = results_df.dropna(subset=['win_prob_change'])
    print("モメンタム計算用データ準備完了。")
    return momentum_df

def calculate_momentum_tables(momentum_df: pd.DataFrame, event_map: dict) -> tuple:
    """イベント別モメンタムの集計表を計算する"""
    print("イベント別モメンタム集計表を計算中...")
    
    # --- ホームチームのイベント ---
    home_events = momentum_df[momentum_df['home_event_id'] != 0].copy()
    home_momentum = home_events.groupby('home_event_id')['win_prob_change'].agg(['mean', 'count']).reset_index()
    home_momentum.rename(columns={'mean': 'avg_win_prob_change', 'count': 'event_count', 'home_event_id': 'event_id'}, inplace=True)
    home_momentum['event_name'] = home_momentum['event_id'].map(event_map).fillna('Unknown Event')
    
    # --- アウェイチームのイベント ---
    visitor_events = momentum_df[momentum_df['visitor_event_id'] != 0].copy()
    visitor_momentum = visitor_events.groupby('visitor_event_id')['win_prob_change'].agg(['mean', 'count']).reset_index()
    visitor_momentum.rename(columns={'mean': 'avg_win_prob_change', 'count': 'event_count', 'visitor_event_id': 'event_id'}, inplace=True)
    visitor_momentum['event_name'] = visitor_momentum['event_id'].map(event_map).fillna('Unknown Event')
    
    print("集計完了。")
    return home_momentum, visitor_momentum

def display_momentum_rankings(home_momentum: pd.DataFrame, visitor_momentum: pd.DataFrame):
    """計算されたモメンタムランキングを表示する"""
    
    home_momentum_sorted = home_momentum.sort_values(by='avg_win_prob_change', ascending=False)
    print("\n--- 【ホームチームのイベント】が勝率に与える影響ランキング ---")
    display_cols = ['event_name', 'avg_win_prob_change', 'event_count', 'event_id']
    with pd.option_context('display.max_rows', None):
        print(home_momentum_sorted[display_cols].to_string())

    visitor_momentum_sorted = visitor_momentum.sort_values(by='avg_win_prob_change', ascending=False)
    print("\n--- 【アウェイチームのイベント】がホームチームの勝率に与える影響ランキング ---")
    with pd.option_context('display.max_rows', None):
        print(visitor_momentum_sorted[display_cols].to_string())

def validate_cumulative_momentum(test_df: pd.DataFrame, home_momentum: pd.DataFrame, visitor_momentum: pd.DataFrame):
    """
    試合ごとにモメンタムを累積し、最終的な累積値の正負と実際の勝敗を比較して精度を検証する
    """
    print("\n--- 累積モメンタムによる勝敗予測の検証 ---")

    # イベントIDをキー、モメンタム値をバリューとする辞書を作成（高速なルックアップのため）
    home_momentum_dict = home_momentum.set_index('event_id')['avg_win_prob_change'].to_dict()
    visitor_momentum_dict = visitor_momentum.set_index('event_id')['avg_win_prob_change'].to_dict()

    game_results = []
    test_game_ids = test_df['game_id'].unique()

    print(f"テストセット内の {len(test_game_ids)} 試合を検証します...")
    for game_id in test_game_ids:
        game_df = test_df[test_df['game_id'] == game_id]
        
        cumulative_momentum = 0
        
        # 各プレーのモメンタムを加算していく
        # ここでのモメンタムは全て「ホームチームの勝率」への影響として計算されているため、
        # 全てを足し合わせると、その試合のホームチームの最終的なモメンタム累積値になる
        home_events_momentum = game_df['home_event_id'].map(home_momentum_dict).fillna(0).sum()
        visitor_events_momentum = game_df['visitor_event_id'].map(visitor_momentum_dict).fillna(0).sum()
        
        cumulative_momentum = home_events_momentum + visitor_events_momentum
        
        actual_winner = game_df['home_win'].iloc[0]
        
        game_results.append({
            'game_id': game_id,
            'final_momentum': cumulative_momentum,
            'actual_home_win': actual_winner
        })

    validation_df = pd.DataFrame(game_results)
    
    # 累積モメンタムの正負で勝敗を予測
    validation_df['predicted_home_win'] = (validation_df['final_momentum'] > 0).astype(int)

    # 精度の計算と表示
    accuracy = accuracy_score(validation_df['actual_home_win'], validation_df['predicted_home_win'])
    
    print("\n--- 検証結果 ---")
    print(f"累積モメンタムの符号に基づく勝敗予測の正解率 (Accuracy): {accuracy:.4f}")
    
    print("\n詳細レポート:")
    print(classification_report(validation_df['actual_home_win'], validation_df['predicted_home_win'], target_names=['Home Loss', 'Home Win']))

    print("\n混同行列 (Confusion Matrix):")
    print("TN FP")
    print("FN TP")
    cm = confusion_matrix(validation_df['actual_home_win'], validation_df['predicted_home_win'])
    print(cm)

    print("\n結果の内訳（先頭10試合）:")
    print(validation_df.head(10).to_string())

def validate_cumulative_momentum_direct(momentum_df: pd.DataFrame):
    """
    プレーごとの「実際の」勝率変動を直接累積し、勝敗予測の精度を検証する（より直接的な方法）
    """
    print("\n--- 【直接検証】累積勝率変動による勝敗予測の検証 ---")

    # game_id ごとに win_prob_change を合計する
    game_results = momentum_df.groupby('game_id').agg(
        final_win_prob_change=('win_prob_change', 'sum'),
        actual_home_win=('home_win', 'first') # 各ゲームの勝敗は同じなので先頭の値を取得
    ).reset_index()

    print(f"テストセット内の {len(game_results)} 試合を検証します...")

    # 累積勝率変動の正負で勝敗を予測
    game_results['predicted_home_win'] = (game_results['final_win_prob_change'] > 0).astype(int)

    # 精度の計算と表示
    accuracy = accuracy_score(game_results['actual_home_win'], game_results['predicted_home_win'])
    
    print("\n--- 検証結果 ---")
    print(f"累積勝率変動の符号に基づく勝敗予測の正解率 (Accuracy): {accuracy:.4f}")
    
    print("\n詳細レポート:")
    print(classification_report(game_results['actual_home_win'], game_results['predicted_home_win'], target_names=['Home Loss', 'Home Win']))

    print("\n混同行列 (Confusion Matrix):")
    cm = confusion_matrix(game_results['actual_home_win'], game_results['predicted_home_win'])
    print(cm)

    print("\n結果の内訳（先頭10試合）:")
    # カラム名を調整して表示
    display_df = game_results.rename(columns={'final_win_prob_change': 'final_momentum'})
    print(display_df.head(10).to_string())

def validate_average_momentum_lookup(test_df: pd.DataFrame, home_momentum: pd.DataFrame, visitor_momentum: pd.DataFrame):
    """
    【平均値ルックアップ検証】
    イベントごとの「平均」モメンタムスコアを累積し、勝敗予測の精度を検証する。
    """
    print("\n--- 【平均値ルックアップ検証】累積モメンタムによる勝敗予測の検証 ---")

    # イベントIDをキー、平均モメンタム値をバリューとする辞書を作成
    home_momentum_dict = home_momentum.set_index('event_id')['avg_win_prob_change'].to_dict()
    visitor_momentum_dict = visitor_momentum.set_index('event_id')['avg_win_prob_change'].to_dict()

    game_results = []
    test_game_ids = test_df['game_id'].unique()

    print(f"テストセット内の {len(test_game_ids)} 試合を検証します...")
    for game_id in test_game_ids:
        game_df = test_df[test_df['game_id'] == game_id].copy()
        
        # 各プレーに対応する「平均」モメンタムスコアをマッピング
        game_df['home_momentum_score'] = game_df['home_event_id'].map(home_momentum_dict).fillna(0)
        game_df['visitor_momentum_score'] = game_df['visitor_event_id'].map(visitor_momentum_dict).fillna(0)
        
        # ホームチーム視点での総モメンタムを計算
        cumulative_momentum = game_df['home_momentum_score'].sum() + game_df['visitor_momentum_score'].sum()
        
        actual_winner = game_df['home_win'].iloc[0]
        
        game_results.append({
            'game_id': game_id,
            'final_momentum_avg': cumulative_momentum,
            'actual_home_win': actual_winner
        })

    validation_df = pd.DataFrame(game_results)
    
    # 累積モメンタムの正負で勝敗を予測
    validation_df['predicted_home_win'] = (validation_df['final_momentum_avg'] > 0).astype(int)

    # 精度の計算と表示
    accuracy = accuracy_score(validation_df['actual_home_win'], validation_df['predicted_home_win'])
    
    print("\n--- 検証結果 ---")
    print(f"平均モメンタムスコアの累積に基づく勝敗予測の正解率 (Accuracy): {accuracy:.4f}")
    
    print("\n詳細レポート:")
    print(classification_report(validation_df['actual_home_win'], validation_df['predicted_home_win'], target_names=['Home Loss', 'Home Win']))

    print("\n混同行列 (Confusion Matrix):")
    cm = confusion_matrix(validation_df['actual_home_win'], validation_df['predicted_home_win'])
    print(cm)