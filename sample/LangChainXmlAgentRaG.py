import sys
import boto3
from langchain.llms import Bedrock
from langchain.chains import LLMChain
from langchain.agents import XMLAgent
from langchain.agents.agent import AgentExecutor
from langchain.agents import Tool
from langchain.retrievers import AmazonKendraRetriever
import sys
sys.path.append('/home/subaru/RaG')
from config import KENDRA_INDEX_ID

# デフォルトのセッションを設定
boto3.setup_default_session(profile_name="develop")

# Amazon Kendraから情報を取得する
# 日本語で"登録されている"ドキュメントを20件(top_k=20)検索して取得結果を全て結合する
kendra_index_id=KENDRA_INDEX_ID #Kendra Index ID
docs=[]
def get_retrieval_result(query) -> str:
    attribute_filter = {"EqualsTo": {"Key": "_language_code","Value": {"StringValue": "ja"}}}
    retriever = AmazonKendraRetriever(credentials_profile_name="develop",
                                      index_id=kendra_index_id,
                                      attribute_filter=attribute_filter,
                                      top_k=20)
    global docs
    docs = retriever.get_relevant_documents(query=query)
    context=""
    for doc in docs:
        context += doc.page_content
    return context

# 使用可能なツールと説明
tools = [
    Tool(
        name = "KendraSearch",
        func=get_retrieval_result,
        description=""""
            このツールは最新のWeb情報を検索するツールです。引数は検索キーワードです。検索キーワードの例は<examples>の通りです。
            <examples>
                <question>Bedrockについて説明してください</question><検索キーワード>Bedrock</検索キーワード>
                <question>Kendraについて英語で教えてください</question><検索キーワード>Kendra</検索キーワード>
                <question>LangChainはBedrockで使用する事が出来ますか？</question><検索キーワード>LangChain Bedrock</検索キーワード>                
                <question>Amplifyは無料で使用する事が出来ますか？</question><検索キーワード>Amplify</検索キーワード>
                <question>アマゾンとは何か</question><検索キーワード>アマゾン</検索キーワード>
            </examples>"""
    )
]

# LLMの定義
llm = Bedrock(
    credentials_profile_name="develop",
    model_id="anthropic.claude-v2",
    model_kwargs={"max_tokens_to_sample": 1000}
)

# Agentの定義
chain = LLMChain(
    llm=llm,
    prompt=XMLAgent.get_default_prompt(),
    output_parser=XMLAgent.get_default_output_parser()
)
agent = XMLAgent(tools=tools, llm_chain=chain)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

# Agentの実行
answer = agent_executor.invoke({"input": "特に言語の指定が無い場合はあなたは質問に対して日本語で回答します。<question>" + sys.argv[1] + "</question>\
    もし<question>の内容が<observation>に含まれない場合は、「検索結果にありません」と答えてください\n\nAssistant:"})
print(answer['output'])

# Kendraで検索したドキュメントのURLを見たい場合は以下
#print("参考文書：")
#for doc in docs:
#    print(doc.metadata["source"])
