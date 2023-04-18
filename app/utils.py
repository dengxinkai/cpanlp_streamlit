def 中国平安年报查询(input_text):
    pinecone.init(api_key="bd20d2c3-f100-4d24-954b-c17928d1c2da",  # find at app.pinecone.io
                      environment="us-east4-gcp",  # next to api key in console
                      namespace="ZGPA_601318")
    index = pinecone.Index(index_name="kedu")
    a=embeddings.embed_query(input_text)
    www=index.query(vector=a, top_k=3, namespace='ZGPA_601318', include_metadata=True)
    c = [x["metadata"]["text"] for x in www["matches"]]
    return c
    
def 双汇发展年报查询(input_text):
    namespace="ShHFZ_000895"
    pinecone.init(api_key="bd20d2c3-f100-4d24-954b-c17928d1c2da",  # find at app.pinecone.io
                      environment="us-east4-gcp",  # next to api key in console
                      namespace=namespace)
    index = pinecone.Index(index_name="kedu")
    a=embeddings.embed_query(input_text)
    www=index.query(vector=a, top_k=3, namespace=namespace, include_metadata=True)
    c = [x["metadata"]["text"] for x in www["matches"]]
    return c
template3 = """尽量以最少的iterations准确快速给出Final Answer,You have access to the following tools:
    {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do,同时准确快速给出Final Answer
    Action: the action to take, should be one of [{tool_names}],同时准确快速给出Final Answer
    Action Input: the input to the action,同时准确快速给出Final Answer
    Observation: the result of the action,同时准确快速给出Final Answer
    Thought: I now know the final answer,同时准确快速给出Final Answer
    Final Answer: the final answer to the original input question,准确快速给出Final Answer
    All inputs and output tokens are limited to 3800.
    Question: {input}
    {agent_scratchpad}
    都用中文表示，除了格式中的Question:Thought:Action:Action Input:Observation:Thought:Final Answer:,总共处理时间不要超过15秒，总的context token不超过2000token。
    be careful,model's maximum context length is 4000 tokens. messages must result in less than 4000 tokens
    """
