import sys
import boto3
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import BedrockEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma
import sys
sys.path.append('/home/subaru/RaG')
from config import CONNECTION_STRING, COLLECTION_NAME



# デフォルトのセッションを設定
# boto3.setup_default_session(profile_name="develop")
session = boto3.Session(profile_name='develop')

# セッションID（この値をチャット履歴のキーとしてDynamoDBから取得／格納する）
session_id = "3"

# チャット履歴の定義（DynamoDBの"SessionTable"を使用。使用されるキーは"SessionId"）
message_history = DynamoDBChatMessageHistory(
    table_name="SessionTable", 
    session_id=session_id,
    boto3_session=session,
    )

# LLMの定義
llm = Bedrock(
    credentials_profile_name="develop",
    model_id="anthropic.claude-v2",
    model_kwargs={"max_tokens_to_sample": 1000},
)

# Embeddingの定義
embedding_function = BedrockEmbeddings(
    credentials_profile_name="develop",
    model_id="amazon.titan-embed-text-v1"
)

# ChromaDBの定義
persist_path = '/home/subaru/RaG/Embedding/chroma'
persist_client = chromadb.PersistentClient(path=persist_path)
db = Chroma(
    client=persist_client,
    collection_name="vector_store",
    embedding_function=embedding_function,
)
# AuroraDBの定義
# AuroraDBのパブリックアクセスを許可しセキュリティグループの作成が必要
# # PostgreSQLの定義
# db = PGVector(
#     connection_string=CONNECTION_STRING,
#     collection_name=COLLECTION_NAME,
#     embedding_function=embedding_function,
# )

# retrieverを定義
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3},
)

# ritrieverに渡す文を生成
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

# 検索する文字列を返す
def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]

# 最終的な回答を生成するプロンプト
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# chainを定義
chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever # | StrOutputParser()
    ) |
    qa_prompt | llm | StrOutputParser()
)

# 実行
input_text = sys.argv[1]
result = chain.invoke({"chat_history": message_history.messages, "question": input_text})
print(result)


# チャット履歴にユーザーのメッセージとAIのメッセージを追加
message_history.add_user_message(input_text)
message_history.add_ai_message(result)


