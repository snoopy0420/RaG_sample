import sys
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores.pgvector import PGVector
import sys
sys.path.append('/home/subaru/RaG')
from config import CONNECTION_STRING, COLLECTION_NAME


# Embeddingの定義
embedding_function = BedrockEmbeddings(
    credentials_profile_name="develop",
    model_id="amazon.titan-embed-text-v1"
)

# ベクターDBの定義
# PostgreSQLの定義
db = PGVector(
    connection_string=CONNECTION_STRING, # 接続文字列
    collection_name=COLLECTION_NAME, # コレクション名
    embedding_function=embedding_function,
)

# retrieverを定義
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3},
)

# llmを定義
llm = Bedrock(
    credentials_profile_name="develop",
    model_id="anthropic.claude-v2",
    model_kwargs={"max_tokens_to_sample": 1000},
)

# promptを定義
# prompt = hub.pull("rlm/rag-prompt")
template = """Answer the question based only on the following context. 
If the context does not contain the relevant content, please respond with 'No document found'.

<context>{context}</context>

<question>{question}</question>
"""
prompt = ChatPromptTemplate.from_template(template)

# chainを定義
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
answer = chain.invoke(sys.argv[1])
print(answer)

