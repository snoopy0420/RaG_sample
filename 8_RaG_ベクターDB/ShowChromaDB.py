import sqlite3

# SQLiteデータベースへの接続
conn = sqlite3.connect('/home/subaru/RaG/Embedding/chroma/chroma.sqlite3')  # データベース名を適切なものに変更する
cursor = conn.cursor()

try:
    # テーブルの一覧を表示

    # クエリを実行
    # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # cursor.execute("SELECT * FROM embeddings")
    cursor.execute("SELECT id FROM embeddings")
    rows = cursor.fetchall()

    # 取得したデータを表示
    for row in rows:
        print(row)

    # データベースの接続をクローズ
    cursor.close()
    conn.close()

except sqlite3.Error as e:
    print("SQLiteエラー：", e)

# try:
#     # テーブルを表示

#     # クエリを実行
#     cursor.execute("SELECT * FROM embeddings")
#     rows = cursor.fetchall()

#     # 取得したデータを表示
#     for row in rows[:2]:
#         print(row)

#     # データベースの接続をクローズ
#     cursor.close()
#     conn.close()

# except sqlite3.Error as e:
#     print("SQLiteエラー：", e)
