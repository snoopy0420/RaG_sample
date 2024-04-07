import sqlite3
import chromadb
from langchain_community.vectorstores import Chroma

# SQLiteデータベースへの接続
# conn = sqlite3.connect('/home/subaru/RaG/Embedding/chroma/chroma.sqlite3')  # データベース名を適切なものに変更する
# cursor = conn.cursor()

# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# cursor.execute("SELECT id FROM embeddings")
# rows = cursor.fetchall()
# # 取得したデータを表示
# for row in rows:
#     print(row)
# # データベースの接続をクローズ
# cursor.close()
# conn.close()

# ChromaDBの定義
# persist_path = '/home/subaru/RaG/Embedding/chroma'  #場所
# client = chromadb.PersistentClient(path=persist_path)
# db = Chroma(
#     collection_name="vector_store",
#     client=client
# )

# 全データを削除
# print("count before", db._collection.count())
# delete_ids = [str(id) for id in range(1, db._collection.count()+1)]
# db.delete(ids=delete_ids)
# print("count after", db._collection.count())

# print(db._collection.id)
# print(db._collection.get(ids=[delete_ids[0]]))
# print(db._collection.get(ids=["fd6e52cd-3462-45a6-8492-4b2022642d24"]))


persist_path = '/home/subaru/RaG/Embedding/chroma'
collection_name = "vector_store"
client = chromadb.PersistentClient(path=persist_path)
collection = client.get_collection(name=collection_name)

# print(collection.count())
# collection.delete(ids=["176"])
# print(collection.count())

# コレクションの削除
print(client.list_collections())
client.delete_collection(collection_name)
print(client.list_collections())

