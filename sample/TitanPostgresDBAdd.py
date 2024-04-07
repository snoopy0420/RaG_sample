import boto3
from langchain_community.embeddings import BedrockEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import S3FileLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
import sys
sys.path.append('/home/subaru/RaG')
from config import CONNECTION_STRING, S3_BUCKET_NAME, S3_FILE_NAME


# デフォルトのセッションを設定
boto3.setup_default_session(profile_name="develop")

# Embeddingsの定義
embeddings = BedrockEmbeddings(
    credentials_profile_name="develop",
    model_id="amazon.titan-embed-text-v1",
    )

# PostgreSQLの定義
db = PGVector(
    connection_string=CONNECTION_STRING, # 接続文字列
    embedding_function=embeddings
)

# ファイルの読み込み(s3)
loader = S3FileLoader(bucket=S3_BUCKET_NAME, key=S3_FILE_NAME)
# ファイルの読み込み(ローカル)
# loader = PyPDFLoader(file_path="/home/subaru/RaG/data/bedrock-ug.pdf")
# chunk分割方法の定義
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=0,
    strip_whitespace=True,
    separators="\n",
    keep_separator=True,
    )
# ロード＆chunk分割
pages = loader.load_and_split(text_splitter=text_spliter)
print(f"make {len(pages)} chunk") 

# DBに追加
db.add_documents(documents=pages, embedding=embeddings)
