# --- データ設定 ---
DB_PATH = '../nba.sqlite'

#読み込み行数指定
LIMIT_ROWS = 225109 


# --- 分析対象のイベントカテゴリ ---
# これらの名前を基に、特徴量を作成
EVENT_CATEGORIES = [
    '2pt_success',
    '3pt_success',
    'shot_miss',
    'ft_success',
    'rebound',
    'unforced_turnover',
    'steal',
]