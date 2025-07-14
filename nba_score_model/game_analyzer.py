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


def analyze_quarterly_momentum(test_game_ids: list, raw_df: pd.DataFrame, scores_df: pd.DataFrame):
    """
    テストセットの各試合について、クォーターごとの累積モメンタムとスコア差を比較・分析する。
    """
    print("\n--- クォーターごとのモメンタムとスコア差の比較分析 ---")

    # モメンタムスコアをルックアップ用辞書に変換
    momentum_scores = {}
    for _, row in scores_df.iterrows():
        event = row['Event Type']
        momentum_scores[f"home_{event}"] = row['Home Score']
        momentum_scores[f"away_{event}"] = row['Away Score']

    quarterly_results = []

    print(f"{len(test_game_ids)}試合のテストデータを分析します...")
    count=0
    for game_id in test_game_ids:
        game_df = raw_df[raw_df['game_id'] == game_id].copy()
        
        # 試合の最終結果を取得
        final_score_row = game_df[game_df['eventmsgtype'] == 13].sort_values('eventnum').iloc[-1]
        
        # 'score' 列が '100 - 90' のような文字列であることを確認
        if isinstance(final_score_row['score'], str) and ' - ' in final_score_row['score']:
            home_final_score, away_final_score = map(int, final_score_row['score'].split(' - '))
            home_win_final = 1 if home_final_score > away_final_score else 0
        else:
            # 最終スコアが取得できない場合はスキップ
            continue

        # 各プレーのモメンタムを計算
        # プレーごとのモメンタムを正しく計算
        game_df['event_categories'] = game_df.apply(categorize_play, axis=1)
        # 複数のカテゴリを持つプレーを展開
        df_exploded = game_df.explode('event_categories')
        # 個々のカテゴリにスコアをマッピング
        df_exploded['momentum_value_single'] = df_exploded['event_categories'].map(momentum_scores).fillna(0)
        # 元のプレー（インデックス）ごとにスコアを合計
        play_momentum = df_exploded.groupby(df_exploded.index)['momentum_value_single'].sum()
        game_df['momentum_value'] = play_momentum
        game_df['cumulative_momentum'] = game_df['momentum_value'].cumsum()
        
        # スコア差を計算
        game_df['score_diff_raw'] = game_df['scoremargin'].apply(process_score_margin)
        game_df['score_diff_raw'] = game_df['score_diff_raw'].ffill()
        game_df['score_diff'] = game_df['score_diff_raw'] * -1

        # 各クォーター終了時点のデータを抽出
        end_of_quarter_df = game_df[game_df['eventmsgtype'] == 13].copy()
        for _, q_row in end_of_quarter_df.iterrows():
            period = q_row['period']
            # 第4クォーターまでを対象
            if period > 4:
                continue

            quarterly_results.append({
                'game_id': game_id,
                'quarter': period,
                'score_diff': q_row['score_diff'],
                'cumulative_momentum': q_row['cumulative_momentum'],
                'final_home_win': home_win_final
            })

    results_df = pd.DataFrame(quarterly_results)

    # 興味深いケースを抽出：「スコアでは負けているが、モメンタムでは勝っている」状況
    comeback_potential = results_df[
        (results_df['score_diff'] < 0) & (results_df['cumulative_momentum'] > 0)
    ]

    print("\n--- 分析結果 ---")
    print("全クォーター終了時点のスコア差とモメンタム（一部）:")
    print(results_df.head(12).to_string())

    if not comeback_potential.empty:
        print("\n--- ★注目ケース：スコアはビハインドだが、モメンタムはプラスの状況 ---")
        print(comeback_potential.to_string())
        
        # その後、実際に勝利した試合の割合を計算
        won_after_potential = comeback_potential['final_home_win'].sum()
        win_rate = (won_after_potential / len(comeback_potential)) * 100
        print(f"\n上記ケースのうち、最終的に勝利した試合の割合: {win_rate:.2f}% ({won_after_potential} / {len(comeback_potential)})")

    else:
        print("\n--- ★注目ケースに該当する試合はありませんでした ---")