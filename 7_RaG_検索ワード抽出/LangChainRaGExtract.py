from langchain.globals import set_debug
set_debug(False) # debug時はTrue

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

# Retrieve用のプロンプトの定義
prompt_pre = PromptTemplate.from_template("""
    あなたはquestionから、検索ツールへの入力となる検索キーワードを考えます。
    questionに後続処理への指示（例：「説明して」「要約して」）が含まれる場合は取り除きます。
    検索キーワードは文章では無く簡潔な単語で指定します。
    検索キーワードは複数の単語を受け付ける事が出来ます。
    検索キーワードは日本語が標準ですが、ユーザー問い合わせに含まれている英単語はそのまま使用してください。
    回答形式は文字列です。
    <question>{question}</question>
""")

# 回答生成用のプロンプトの定義
prompt_main = PromptTemplate.from_template("""
    あなたはcontextを参考に、questionに回答します。
    <context>{context}</context>
    <question>{question}</question>
""")

# LLMの定義
LLM = Bedrock(
    credentials_profile_name="develop",
    model_id="anthropic.claude-v2",
    model_kwargs={"max_tokens_to_sample": 1000},
)

# Retriever(Kendra)の定義（Kendra Index ID、言語、取得件数）
kendra_index_id=KENDRA_INDEX_ID
retriever = AmazonKendraRetriever(
    credentials_profile_name="develop",
    index_id=kendra_index_id,
    attribute_filter={"EqualsTo": {"Key": "_language_code","Value": {"StringValue": "ja"}}},
    top_k=10
)

# chainの定義
chain = (
    {"context": prompt_pre | LLM | retriever, "question":  RunnablePassthrough()}
    | prompt_main 
    | LLM
)

# chainの実行
answer = chain.invoke({"question": sys.argv[1]})
print(answer)
