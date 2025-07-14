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
    #'2pt_success',
    #'3pt_success',
    'shot_miss_unblocked', # ブロックされなかったシュート失敗
    #'ft_success',
    'ft_miss',            
    'rebound_off',        
    'rebound_def',         
    'unforced_turnover',
    'steal',               
    'block',                          
    'foul',                
]


# --- モデルが使用する特徴量のリスト（ホームとアウェイを分離） ---
MODEL_FEATURES = []
for category in EVENT_CATEGORIES:
    MODEL_FEATURES.append(f'home_{category}')
    MODEL_FEATURES.append(f'away_{category}')