import pandas as pd

# 他のモジュールから必要な関数をインポート
from data_processor import categorize_play
from utils import calculate_seconds_elapsed, process_score_margin

def export_game_analysis_to_excel(game_id: str, raw_df: pd.DataFrame, scores_df: pd.DataFrame):
    """
    指定された1試合のモメンタム推移とスコアを計算し、Excelファイルとして保存する。
    """
    print(f"\n--- 試合 {game_id} の分析データを作成しExcelに出力中 ---")

    # 1. スコアをルックアップしやすい辞書形式に変換
    # ※model.pyで定義した列名と完全に一致させる
    momentum_scores = {}
    for _, row in scores_df.iterrows():
        event = row['Event Type']
        momentum_scores[f"home_{event}"] = row['Home Score']
        momentum_scores[f"away_{event}"] = row['Away Score']

    # 2. 指定された試合のPBPデータだけを抽出・準備
    game_df = raw_df[raw_df['game_id'] == game_id].copy()
    game_df = game_df.sort_values('eventnum').reset_index(drop=True)

    if game_df.empty:
        print(f"エラー: 試合ID {game_id} のデータが見つかりません。")
        return

    # 3a. スコア差の文字列を数値に変換
    game_df['score_diff_raw'] = game_df['scoremargin'].apply(process_score_margin)
    # 3b. 欠損値を前の値で埋める (ffill)
    game_df['score_diff_raw'] = game_df['score_diff_raw'].ffill()
    # 3c. 元データは visitor - home なので、-1を掛けて home - visitor に修正
    #game_df['score_diff'] = game_df['score_diff_raw'] * -1
    game_df['score_diff'] = game_df['score_diff_raw']

    # 3. 各プレーをカテゴリ分類し、モメンタムスコアを割り当て
    game_df['event_category'] = game_df.apply(categorize_play, axis=1)
    game_df['momentum_value'] = game_df['event_category'].map(momentum_scores).fillna(0)
    
    # 4. モメンタムを累積
    game_df['cumulative_momentum'] = game_df['momentum_value'].cumsum()

    # 5. Excelに出力する列を定義し、整理
    output_df = game_df[[
        'eventnum',
        'period',
        'pctimestring',
        'score',  # その時点のスコア
        'score_diff', 
        'homedescription',
        'visitordescription',
        'event_category',
        'momentum_value',      # このプレー単体のモメンタムスコア
        'cumulative_momentum'  # 累積モメンタムスコア
    ]]
    
    # 列名をより分かりやすく変更
    output_df = output_df.rename(columns={
        'eventnum': 'プレー番号',
        'period': 'ピリオド',
        'pctimestring': '残り時間',
        'score': 'スコア',
        'score_diff': 'スコア差',
        'homedescription': 'ホームチームプレー内容',
        'visitordescription': 'アウェイチームプレー内容',
        'event_category': 'イベントカテゴリ',
        'momentum_value': '単体モメンタム',
        'cumulative_momentum': '累積モメンタム'
    })

    # 6. Excelファイルとして保存
    filename = f'game_analysis_{game_id}.xlsx'
    try:
        output_df.to_excel(filename, index=False, engine='openpyxl')
        print(f"分析データを '{filename}' として保存しました。")
    except Exception as e:
        print(f"Excelファイルの保存中にエラーが発生しました: {e}")
        print("`pip install openpyxl` または `conda install openpyxl` を実行したか確認してください。")