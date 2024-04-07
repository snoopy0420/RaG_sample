import sys
from langchain_community.embeddings import BedrockEmbeddings
from langchain.vectorstores.pgvector import PGVector
import sys
sys.path.append('/home/subaru/RaG')
from config import CONNECTION_STRING, COLLECTION_NAME


# Embeddingの定義
embedding_function = BedrockEmbeddings(
    credentials_profile_name="develop",
    model_id="amazon.titan-embed-text-v1"
)

# PostgreSQLの定義
db = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function,
)

# 類似度検索
query = "ベースモデル"
answer = db.similarity_search_with_score(
    query=query,
    k=4,
    )
print(answer)