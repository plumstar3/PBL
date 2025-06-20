import pandas as pd
from typing import Optional

# --- Helper Functions ---
def parse_time_to_seconds(time_str: str) -> Optional[int]:
    if isinstance(time_str, str) and ':' in time_str:
        try:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds
        except ValueError:
            return None
    return None

def calculate_seconds_elapsed(row: pd.Series) -> Optional[float]:
    period = row['period']
    pctimestring = row['pctimestring']
    if pd.isna(period) or period < 1: return None
    seconds_in_period = parse_time_to_seconds(pctimestring)
    if seconds_in_period is None: return None
    seconds_per_period = 720 if period <= 4 else 300
    seconds_elapsed_in_current_period = seconds_per_period - seconds_in_period
    if not (0 <= seconds_elapsed_in_current_period <= seconds_per_period): return None
    if period <= 4:
        return (period - 1) * 720 + seconds_elapsed_in_current_period
    else:
        return 4 * 720 + (period - 5) * 300 + seconds_elapsed_in_current_period

def process_score_margin(margin_str: str) -> Optional[int]:
    if margin_str == 'TIE': return 0
    if pd.isna(margin_str) or margin_str == '': return None
    try:
        return int(str(margin_str).replace('+', ''))
    except ValueError:
        return None

def assign_contextual_event_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    descriptionの内容を解析し、イベントの真の主体を判断して
    home_event_id と visitor_event_id を割り当てる。
    """
    print("文脈を考慮したイベントIDの割り当てを開始...")
    home_desc = df['homedescription'].fillna('').str.upper()
    visit_desc = df['visitordescription'].fillna('').str.upper()
    base_event_id = (df['eventmsgtype'] * 1000 + df['eventmsgactiontype']).fillna(0).astype(int)

    home_ids = pd.Series(0, index=df.index)
    visitor_ids = pd.Series(0, index=df.index)
    
    # ルールベースでの割り当て（元のコードをそのまま利用）
    # (元のコードのセクション5にある assign_contextual_event_ids 関数のロジックをここにコピー)
    # ...
    is_shot_or_ft = df['eventmsgtype'].isin([1, 2, 3])
    home_action_shot = is_shot_or_ft & (home_desc != '') & (visit_desc == '')
    visitor_action_shot = is_shot_or_ft & (visit_desc != '') & (home_desc == '')
    home_ids.loc[home_action_shot] = base_event_id[home_action_shot]
    visitor_ids.loc[visitor_action_shot] = base_event_id[visitor_action_shot]

    is_rebound = df['eventmsgtype'] == 4
    home_rebound = is_rebound & home_desc.str.contains("REBOUND")
    visitor_rebound = is_rebound & visit_desc.str.contains("REBOUND")
    home_ids.loc[home_rebound] = base_event_id[home_rebound]
    visitor_ids.loc[visitor_rebound] = base_event_id[visitor_rebound]

    is_turnover = df['eventmsgtype'] == 5
    home_steal = is_turnover & home_desc.str.contains("STEAL")
    visitor_steal = is_turnover & visit_desc.str.contains("STEAL")
    visitor_ids.loc[home_steal] = base_event_id[home_steal]
    home_ids.loc[visitor_steal] = base_event_id[visitor_steal]

    no_steal_turnover = is_turnover & ~home_steal & ~visitor_steal
    home_turnover_no_steal = no_steal_turnover & home_desc.str.contains("TURNOVER")
    visitor_turnover_no_steal = no_steal_turnover & visit_desc.str.contains("TURNOVER")
    home_ids.loc[home_turnover_no_steal] = base_event_id[home_turnover_no_steal]
    visitor_ids.loc[visitor_turnover_no_steal] = base_event_id[visitor_turnover_no_steal]

    is_foul = df['eventmsgtype'] == 6
    home_foul = is_foul & home_desc.str.contains("FOUL")
    visitor_foul = is_foul & visit_desc.str.contains("FOUL")
    home_ids.loc[home_foul] = base_event_id[home_foul]
    visitor_ids.loc[visitor_foul] = base_event_id[visitor_foul]

    is_other_event = ~df['eventmsgtype'].isin([1,2,3,4,5,6])
    home_other = is_other_event & (home_desc != '')
    visitor_other = is_other_event & (visit_desc != '')
    home_ids.loc[home_other] = base_event_id[home_other]
    visitor_ids.loc[visitor_other] = base_event_id[visitor_other]
    
    unassigned = (home_ids == 0) & (visitor_ids == 0)
    home_ids.loc[unassigned & (home_desc != '') & (visit_desc == '')] = base_event_id[unassigned & (home_desc != '') & (visit_desc == '')]
    visitor_ids.loc[unassigned & (visit_desc != '') & (home_desc == '')] = base_event_id[unassigned & (visit_desc != '') & (home_desc == '')]

    df['home_event_id'] = home_ids
    df['visitor_event_id'] = visitor_ids
    df['composite_event_id'] = base_event_id

    print("イベントIDの割り当て完了。")
    return df

# --- Main Feature Creation Function ---
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    データフレームを受け取り、分析に必要なすべての特徴量を生成する。
    """
    print("\n--- 特徴量エンジニアリング開始 ---")
    
    df_featured = df.copy()

    print("総経過時間を計算中...")
    df_featured['seconds_elapsed'] = df_featured.apply(calculate_seconds_elapsed, axis=1)

    print("得点差を処理中...")
    df_featured['numeric_score_margin'] = df_featured['scoremargin'].apply(process_score_margin)
    df_featured = df_featured.sort_values(by=['game_id', 'eventnum'])
    df_featured['numeric_score_margin'] = df_featured.groupby('game_id')['numeric_score_margin'].ffill()
    
    print("得点差の正負を標準的な表記 (home - visitor) に修正中...")
    df_featured['numeric_score_margin'] = df_featured['numeric_score_margin'] * -1

    df_featured = assign_contextual_event_ids(df_featured)

    print("--- 特徴量エンジニアリング完了 ---")
    return df_featured