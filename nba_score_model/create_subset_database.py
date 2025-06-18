# create_subset_database.py

import sqlite3
import pandas as pd
import os

# --- 設定 ---
# 元の巨大なデータベースのパス（一つ上の階層）
ORIGINAL_DB_PATH = '../nba.sqlite'

# 新しく作成する小さなデータベースのパスと名前
NEW_DB_PATH = 'nba_small.sqlite'

# 抜き出すテーブル名
TABLE_NAME = 'play_by_play'

# 抜き出す行数
ROW_LIMIT = 225109

def create_subset_db():
    """
    巨大なSQLiteデータベースから指定した行数だけを抜き出し、
    新しい小さなデータベースファイルを作成する。
    """
    print("--- データベースのサブセット作成開始 ---")

    # 1. 元の巨大なデータベースからデータを読み込む
    if not os.path.exists(ORIGINAL_DB_PATH):
        print(f"エラー: 元となるデータベースが見つかりません: {ORIGINAL_DB_PATH}")
        return

    try:
        print(f"元のデータベースに接続中: {ORIGINAL_DB_PATH}")
        conn_orig = sqlite3.connect(ORIGINAL_DB_PATH)

        query = f"SELECT * FROM {TABLE_NAME} LIMIT {ROW_LIMIT}"

        print(f"テーブル '{TABLE_NAME}' から先頭 {ROW_LIMIT} 行を読み込み中...")
        df = pd.read_sql_query(query, conn_orig)

        conn_orig.close()
        print(f"読み込み完了。{len(df)}行を取得しました。")

    except Exception as e:
        print(f"データベースの読み込み中にエラーが発生しました: {e}")
        return

    # 2. 新しい小さなデータベースにデータを書き込む
    try:
        print(f"新しいデータベースに接続中: {NEW_DB_PATH}")
        conn_new = sqlite3.connect(NEW_DB_PATH)

        print(f"テーブル '{TABLE_NAME}' に {len(df)} 行を書き込み中...")
        # if_exists='replace' は、もし同名のテーブルが既に存在した場合、それを削除して作り直す設定
        # index=False は、pandasのインデックスがデータベースに列として保存されるのを防ぐ設定
        df.to_sql(TABLE_NAME, conn_new, if_exists='replace', index=False)

        conn_new.close()
        print("書き込み完了。")

    except Exception as e:
        print(f"新しいデータベースへの書き込み中にエラーが発生しました: {e}")
        return

    print(f"\n成功: '{NEW_DB_PATH}' が作成されました。")
    print("--- 処理終了 ---")


if __name__ == "__main__":
    create_subset_db()