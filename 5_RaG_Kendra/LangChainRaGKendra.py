import sys
import boto3
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Bedrock
from langchain_community.retrievers import AmazonKendraRetriever
from langchain_core.runnables import RunnablePassthrough
import sys
sys.path.append('/home/subaru/RaG')
from config import KENDRA_INDEX_ID

# デフォルトのセッションを設定
boto3.setup_default_session(profile_name="develop")

# Retriever(Kendra)の定義
# 日本語で"登録されている"ドキュメントを20件(top_k=20)検索する、と定義
kendra_index_id=KENDRA_INDEX_ID #Kendra Index ID
attribute_filter = {"EqualsTo": {"Key": "_language_code","Value": {"StringValue": "ja"}}}
retriever = AmazonKendraRetriever(
    credentials_profile_name="develop",
    index_id=kendra_index_id,
    attribute_filter=attribute_filter,
    top_k=10
)

# LLMの定義
llm = Bedrock(
    credentials_profile_name="develop",
    model_id="anthropic.claude-v2",
    model_kwargs={"max_tokens_to_sample": 1000}
)

# promptの定義
prompt_template = """
  <documents>タグには参考文書が書かれています。
  <documents>{context}</documents>
  \n\nHuman: 上記参考文書を元に、<question>に対して説明してください。言語の指定が無い場合は日本語で答えてください。
    もし<question>の内容が参考文書に無かった場合は「文書にありません」と答えてください。
  <question>{question}</question>
  \n\nAssistant:"""
prompt = PromptTemplate.from_template(template=prompt_template)

# chainの定義
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

# chainの実行
result = chain.invoke(sys.argv[1])
print(result)
