import pandas as pd

def analyze_momentum(results_df: pd.DataFrame, event_map: dict):
    """
    予測結果のデータフレームからイベント別のモメンタムを計算し、表示する
    """
    print("\n--- イベント別モメンタム分析 ---")
    
    results_df = results_df.sort_values(by=['game_id', 'eventnum'])
    results_df['prev_win_prob'] = results_df.groupby('game_id')['win_probability_pred'].shift(1)
    results_df['win_prob_change'] = results_df['win_probability_pred'] - results_df['prev_win_prob']
    momentum_df = results_df.dropna(subset=['win_prob_change'])

    # --- 1. ホームチームのイベント分析 ---
    home_events = momentum_df[momentum_df['home_event_id'] != 0].copy()
    if not home_events.empty:
        home_momentum = home_events.groupby('home_event_id')['win_prob_change'].agg(['mean', 'count']).reset_index()
        home_momentum.rename(columns={'mean': 'avg_win_prob_change', 'count': 'event_count', 'home_event_id': 'event_id'}, inplace=True)
        home_momentum['event_name'] = home_momentum['event_id'].map(event_map).fillna('Unknown Event')
        home_momentum_sorted = home_momentum.sort_values(by='avg_win_prob_change', ascending=False)
        
        print("\n--- 【ホームチームのイベント】が勝率に与える影響ランキング ---")
        display_cols = ['event_name', 'avg_win_prob_change', 'event_count', 'event_id']
        with pd.option_context('display.max_rows', None):
            print(home_momentum_sorted[display_cols].to_string())

    # --- 2. アウェイチームのイベント分析 ---
    visitor_events = momentum_df[momentum_df['visitor_event_id'] != 0].copy()
    if not visitor_events.empty:
        visitor_momentum = visitor_events.groupby('visitor_event_id')['win_prob_change'].agg(['mean', 'count']).reset_index()
        visitor_momentum.rename(columns={'mean': 'avg_win_prob_change', 'count': 'event_count', 'visitor_event_id': 'event_id'}, inplace=True)
        visitor_momentum['event_name'] = visitor_momentum['event_id'].map(event_map).fillna('Unknown Event')
        visitor_momentum_sorted = visitor_momentum.sort_values(by='avg_win_prob_change', ascending=False)
        
        print("\n--- 【アウェイチームのイベント】がホームチームの勝率に与える影響ランキング ---")
        display_cols = ['event_name', 'avg_win_prob_change', 'event_count', 'event_id']
        with pd.option_context('display.max_rows', None):
            print(visitor_momentum_sorted[display_cols].to_string())