# --- データ設定 ---
DB_PATH = '../nba.sqlite'

#読み込み行数指定
#500試合
LIMIT_ROWS = 225109 
#2440試合
#LIMIT_ROWS = 1100917
#4410試合
#LIMIT_ROWS = 2000600


# --- 分析対象のイベントカテゴリ ---
# これらの名前を基に、特徴量を作成
EVENT_CATEGORIES = [
    'shot_miss_unblocked',
    'ft_miss',
    'rebound',
    'unforced_turnover',
    'steal',
    'block',
    'foul',
    # '2pt_success',
    # '3pt_success',
    # 'ft_success',
]


# --- モデルが使用する特徴量のリスト（ホームとアウェイを分離,クオーター別に分離） ---
MODEL_FEATURES = []
QUARTERS = [1, 2, 3, 4]

for category in EVENT_CATEGORIES:
    for q in QUARTERS:
        MODEL_FEATURES.append(f'home_{q}q_{category}')
        MODEL_FEATURES.append(f'away_{q}q_{category}')