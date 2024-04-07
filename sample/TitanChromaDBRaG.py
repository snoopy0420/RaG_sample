import sys
import boto3
from langchain_community.embeddings import BedrockEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Bedrock
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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

# retrieverを定義
retriever = db.as_retriever(
    serch_type="similarity",
    search_kwargs={"k":3} # 検索する文書チャンク数
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

# chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
answer = chain.invoke(sys.argv[1])
print(answer)