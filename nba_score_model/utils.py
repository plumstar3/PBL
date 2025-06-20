from typing import Optional
import pandas as pd

def parse_time_to_seconds(time_str: str) -> Optional[int]:
    """'MM:SS' 形式の文字列を秒に変換する"""
    if isinstance(time_str, str) and ':' in time_str:
        try:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds
        except ValueError:
            return None
    return None

def calculate_seconds_elapsed(row: pd.Series) -> Optional[float]:
    """試合開始からの経過時間（秒）を計算する"""
    period = row['period']
    pctimestring = row['pctimestring']
    if pd.isna(period) or period < 1:
        return None
        
    seconds_in_period = parse_time_to_seconds(pctimestring)
    if seconds_in_period is None:
        return None
        
    seconds_per_period = 720 if period <= 4 else 300  # 12分 or 5分
    seconds_elapsed_in_current_period = seconds_per_period - seconds_in_period
    
    if not (0 <= seconds_elapsed_in_current_period <= seconds_per_period):
        return None
        
    if period <= 4:
        total_seconds_elapsed = (period - 1) * 720 + seconds_elapsed_in_current_period
    else: # 延長ピリオド
        total_seconds_elapsed = (4 * 720) + (period - 5) * 300 + seconds_elapsed_in_current_period
        
    return total_seconds_elapsed

def process_score_margin(margin_str: str) -> Optional[int]:
    """
    得点差の文字列を数値に変換する。
    """
    if margin_str == 'TIE':
        return 0
    if pd.isna(margin_str) or margin_str == '':
        return None
    try:
        # "+"記号などを除去して数値に変換
        return int(str(margin_str).replace('+', ''))
    except ValueError:
        return None