# game_analyzer.py

import pandas as pd
from data_processor import categorize_play
from utils import process_score_margin

import statsmodels.api as sm
import statsmodels.formula.api as smf


def export_game_analysis_to_excel(game_id: str, raw_df: pd.DataFrame, scores_df: pd.DataFrame):
    """
    指定された1試合のモメンタム推移とスコアを計算し、Excelファイルとして保存する。
    """
    print(f"\n--- 試合 {game_id} の分析データを作成しExcelに出力中 ---")

    momentum_scores = {}
    for _, row in scores_df.iterrows():
        event = row['Event Type']
        momentum_scores[f"home_{event}"] = row['Home Score']
        momentum_scores[f"away_{event}"] = row['Away Score']

    game_df = raw_df[raw_df['game_id'] == game_id].copy()
    game_df = game_df.sort_values('eventnum').reset_index(drop=True)

    if game_df.empty:
        print(f"エラー: 試合ID {game_id} のデータが見つかりません。")
        return

    game_df['score_diff_raw'] = game_df['scoremargin'].apply(process_score_margin)
    game_df['score_diff_raw'] = game_df['score_diff_raw'].ffill()
    game_df['score_diff'] = game_df['score_diff_raw'] * -1
    
    # --- ▼▼▼ ここからが今回のエラーの修正箇所 ▼▼▼ ---
    # プレーごとのモメンタムを正しく計算
    game_df['event_categories'] = game_df.apply(categorize_play, axis=1)
    df_exploded = game_df.explode('event_categories')
    df_exploded['momentum_value_single'] = df_exploded['event_categories'].map(momentum_scores).fillna(0)
    # 元のプレー（インデックス）ごとにスコアを合計
    play_momentum = df_exploded.groupby(df_exploded.index)['momentum_value_single'].sum()
    game_df['momentum_value'] = play_momentum
    # --- ▲▲▲ 修正ここまで ▲▲▲ ---
    
    game_df['cumulative_momentum'] = game_df['momentum_value'].cumsum()

    output_df = game_df[[
        'eventnum', 'period', 'pctimestring', 'score', 'score_diff',
        'homedescription', 'visitordescription', 'event_categories',
        'momentum_value', 'cumulative_momentum'
    ]].rename(columns={'event_categories': 'event_category'})
    
    output_df = output_df.rename(columns={
        'eventnum': 'プレー番号', 'period': 'ピリオド', 'pctimestring': '残り時間',
        'score': 'スコア', 'score_diff': 'スコア差',
        'homedescription': 'ホームチームプレー内容', 'visitordescription': 'アウェイチームプレー内容',
        'event_category': 'イベントカテゴリ', 'momentum_value': '単体モメンタム',
        'cumulative_momentum': '累積モメンタム'
    })

    filename = f'game_analysis_{game_id}.xlsx'
    try:
        output_df.to_excel(filename, index=False, engine='openpyxl')
        print(f"分析データを '{filename}' として保存しました。")
    except Exception as e:
        print(f"Excelファイルの保存中にエラーが発生しました: {e}")

def create_quarterly_data(test_game_ids: list, raw_df: pd.DataFrame, scores_df: pd.DataFrame):
    """
    テストセットの各試合について、クォーターごとの累積モメンタムとスコア差を比較・分析する。
    """
    print("\n--- クォーターごとのモメンタムとスコア差の比較分析 ---")

    momentum_scores = {}
    for _, row in scores_df.iterrows():
        event = row['Event Type']
        momentum_scores[f"home_{event}"] = row['Home Score']
        momentum_scores[f"away_{event}"] = row['Away Score']

    quarterly_results = []

    print(f"{len(test_game_ids)}試合のテストデータを分析します...")
    for game_id in test_game_ids:
        game_df = raw_df[raw_df['game_id'] == game_id].copy()
        
        final_score_row = game_df[game_df['eventmsgtype'] == 13].sort_values('eventnum').iloc[-1]
        
        if isinstance(final_score_row['score'], str) and ' - ' in final_score_row['score']:
            # スコアの順序を正しく解釈 (away, home)
            away_final_score, home_final_score = map(int, final_score_row['score'].split(' - '))
            home_win_final = 1 if home_final_score > away_final_score else 0
        else:
            continue

        game_df['event_categories'] = game_df.apply(categorize_play, axis=1)
        df_exploded = game_df.explode('event_categories')
        df_exploded['momentum_value_single'] = df_exploded['event_categories'].map(momentum_scores).fillna(0)
        play_momentum = df_exploded.groupby(df_exploded.index)['momentum_value_single'].sum()
        game_df['momentum_value'] = play_momentum
        
        game_df['cumulative_momentum'] = game_df['momentum_value'].cumsum()
        
        game_df['score_diff_raw'] = game_df['scoremargin'].apply(process_score_margin)
        game_df['score_diff_raw'] = game_df['score_diff_raw'].ffill()
        game_df['score_diff'] = game_df['score_diff_raw']

        end_of_quarter_df = game_df[game_df['eventmsgtype'] == 13].copy()
        for _, q_row in end_of_quarter_df.iterrows():
            period = q_row['period']
            if period > 3:
                continue

            quarterly_results.append({
                'game_id': game_id,
                'quarter': int(period),
                'score_diff': q_row['score_diff'],
                'cumulative_momentum': q_row['cumulative_momentum'],
                'final_home_win': home_win_final
            })

    if not quarterly_results:
        print("分析対象のクォーターデータがありませんでした。")
        return pd.DataFrame()
    
    return pd.DataFrame(quarterly_results)
        
def analyze_comeback_cases(quarterly_df: pd.DataFrame):
    """
    クォーターごとのデータから、逆転の可能性に関する比較分析を行う。
    """
    print("\n--- 逆転可能性の比較分析 ---")
    
    all_losing_situations = quarterly_df[quarterly_df['score_diff'] < 0]
    comeback_potential = all_losing_situations[all_losing_situations['cumulative_momentum'] > 0]

    if not all_losing_situations.empty:
        won_after_losing = all_losing_situations['final_home_win'].sum()
        win_rate_losing = (won_after_losing / len(all_losing_situations)) * 100
        print(f"\n[比較対象] スコアがビハインドだった全状況の場合:")
        print(f"  -> 最終的な勝率: {win_rate_losing:.2f}% ({won_after_losing}勝 / {len(all_losing_situations)}状況)")
    else:
        print("\n[比較対象] スコアがビハインドの状況はありませんでした。")

    if not comeback_potential.empty:
        won_after_potential = comeback_potential['final_home_win'].sum()
        win_rate_potential = (won_after_potential / len(comeback_potential)) * 100
        print(f"\n[注目ケース] うち、モメンタムがプラスだった場合:")
        print(f"  -> 最終的な勝率: {win_rate_potential:.2f}% ({won_after_potential}勝 / {len(comeback_potential)}状況)")
    else:
        print("\n[注目ケース] に該当する状況はありませんでした。")

def run_gee_model(quarterly_results_df: pd.DataFrame):
    """
    クォーターごとのデータを用いて、GEEモデルを実行する。
    """
    print("\n--- GEEモデル（一般化推定方程式）による分析 ---")
    
    df = quarterly_results_df.dropna(subset=['final_home_win', 'cumulative_momentum', 'score_diff', 'game_id'])

    if len(df) < 2:
        print("分析に十分なデータがありません。")
        return

    try:
        # GEEモデルを定義
        # groups=df["game_id"] で、game_idごとにデータがグループ化されていることを指定
        # family=sm.families.Binomial() で、ロジスティック回帰と同じ種類の分析を指定
        model = smf.gee("final_home_win ~ cumulative_momentum + score_diff",
                        groups=df["game_id"], data=df, family=sm.families.Binomial())
        
        result = model.fit()
        
        print("\n--- GEEモデル分析結果サマリー ---")
        print(result.summary())

    except Exception as e:
        print(f"GEEモデルの実行中にエラーが発生しました: {e}")