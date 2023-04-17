import streamlit as st
import numpy as np
import pandas as pd
import base64
import json
import os
import tempfile
import pinecone 
import requests
import re
from typing import List, Union,Callable
from langchain.agents import  AgentExecutor, LLMSingleActionAgent, AgentOutputParser,initialize_agent, Tool,AgentType,create_pandas_dataframe_agent
from langchain.prompts import StringPromptTemplate,PromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.schema import AgentAction, AgentFinish,Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma,FAISS
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper,WikipediaAPIWrapper,TextRequestsWrapper
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
st.set_page_config(
    page_title="可读-财报GPT",
    page_icon="https://raw.githubusercontent.com/dengxinkai/cpanlp_streamlit/main/app/%E6%9C%AA%E5%91%BD%E5%90%8D.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.cpanlp.com/',
        'Report a bug': "https://www.cpanlp.com/",
        'About': "可读-财报GPT"
    }
)
with st.sidebar:
    st.header(":blue[Openai_api]")
    st.text_input('api-key', '', key="input_api")
    st.info('为防止bug,请正确输入您的openai的apikey', icon="😊")
    
st.title('中国上市公司智能财报阅读')
if st.session_state.input_api:
    @st.cache(allow_output_mutation=True)
    def getseccode(text):
        pattern = r"\d{6}"
        result=re.findall(pattern, text)
        return result[0]

    def gettoken(client_id,client_secret):
        url='http://webapi.cninfo.com.cn/api-cloud-platform/oauth2/token'
        post_data="grant_type=client_credentials&client_id=%s&client_secret=%s"%(client_id,client_secret)
        post_data={"grant_type":"client_credentials",
                   "client_id":client_id,
                   "client_secret":client_secret
                   }
        req = requests.post(url, data=post_data)
        tokendic = json.loads(req.text)
        return tokendic['access_token']
    @st.cache(allow_output_mutation=True)
    def cnifo(scode):
        token = gettoken('TvUN4uIl2gu4sjPdB4su6DiPNYFMkhA1','Snb5s887ezAWBXIyYyqY5fBQI6ttyySu')
    #上市公司市场数据
        url = 'http://webapi.cninfo.com.cn/api/stock/p_stock2402?&scode={}&edate=20230415&access_token='.format(scode)+token
        requests = TextRequestsWrapper()
        a=requests.get(url)
        data = json.loads(a)
        df = pd.json_normalize(data, record_path='records')
        df=df.rename(columns={'SECNAME': '证券简称', 'F009N': '涨跌','F008N': '总笔数', 'SECCODE': '证券代码','TRADEDATE': '交易日期', 'F001V': '交易所','F002N': '昨日收盘价', 'F003N': '今日开盘价','F004N': '成交数量', 'F005N': '最高成交价',"F006N":"最低成交价","F007N":"最近成交价","F010N":"涨跌幅","F011N":"成交金额","F012N":"换手率","F013N":"振幅","F020N":"发行总股本","F021N":"流通股本","F026N":"市盈率"})
        df['交易日期'] = pd.to_datetime(df['交易日期'])
        agent_df = create_pandas_dataframe_agent(OpenAI(temperature=0.4), df, verbose=True,return_intermediate_steps=True)
        return agent_df 
    @st.cache(allow_output_mutation=True)
    def 中国平安(input_text):
        pinecone.init(api_key="bd20d2c3-f100-4d24-954b-c17928d1c2da",  # find at app.pinecone.io
                          environment="us-east4-gcp",  # next to api key in console
                          namespace="ZGPA_601318")
        index = pinecone.Index(index_name="kedu")
        a=embeddings.embed_query(input_text)
        www=index.query(vector=a, top_k=3, namespace='ZGPA_601318', include_metadata=True)
        c = [x["metadata"]["text"] for x in www["matches"]]
        return c

    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.input_api)
    wikipedia = WikipediaAPIWrapper()
    llm=ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        frequency_penalty=1,
        presence_penalty=1,
        top_p=0.5,
        openai_api_key=st.session_state.input_api
    )
    search = GoogleSearchAPIWrapper(google_api_key="AIzaSyCLKh_M6oShQ6rUJiw8UeQ74M39tlCUa9M",google_cse_id="c147e3f22fbdb4316")
    search_tool =  Tool(
                    name = "Google",
                    func=search.run,
                    description="当您需要回答有关当前财经问题时，这个工具非常有用。"
                )
    zgpa_tool =  Tool(
                    name = "ZGPA",
                    func=中国平安,
                    description="当您需要回答有关中国平安(601318)财报信息的问题时，这个工具非常有用。"
                )
    ALL_TOOLS = [search_tool,zgpa_tool]
    docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(ALL_TOOLS)]
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=st.session_state.input_api))
    retriever = vector_store.as_retriever()
    def get_tools(query):
        docs = retriever.get_relevant_documents(query)
        return [ALL_TOOLS[d.metadata["index"]] for d in docs]
    global qa
    logo_url = "https://raw.githubusercontent.com/dengxinkai/cpanlp_streamlit/main/app/%E6%9C%AA%E5%91%BD%E5%90%8D.png"
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer in Chinese:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    # prompt_template1 = """Write a concise summary of the following:
    # {text}
    # CONCISE SUMMARY IN CHINESE:"""
    # PROMPT1 = PromptTemplate(template=prompt_template1, input_variables=["text"])
    # result = ""
    template3 = """Answer the following questions as best you can.When the question is unrelated to financial matters:
    Final Answer:无法回答，因为与财经无关
    Don't use any tool。
    When the question is related to financial matters，You have access to the following tools:
    {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    All inputs and output tokens are limited to 3800.
    Question: {input}
    {agent_scratchpad}
    最后把输出的Final Answer结果翻译成中文
    """
    # Set up a prompt template
    class CustomPromptTemplate(StringPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools_getter: Callable
        def format(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            tools = self.tools_getter(kwargs["input"])
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
            return self.template.format(**kwargs)
    class CustomPromptTemplate_Upload(StringPromptTemplate):
        # The template to use
        template: str
        # The list of tools available
        tools: List[Tool]
        def format(self, **kwargs) -> str:
            # Get the intermediate steps (AgentAction, Observation tuples)
            # Format them in a particular way
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            return self.template.format(**kwargs)
    class CustomOutputParser(AgentOutputParser):
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            # Check if agent should finish
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            # Parse out the action and action input
            regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    output_parser = CustomOutputParser()


    @st.cache(allow_output_mutation=True)
    def 分析(input_text):
        if file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.read())
                tmp_file.flush()
                loader = PyPDFLoader(tmp_file.name)
        elif input_text != "":        
            loader = PyPDFLoader(input_text)
        else:
            return None
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.input_api)
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever()
        return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)
    qa = 分析(input_text)
    st.header(":blue[上传]")
    file = st.file_uploader("PDF文件", type="pdf")
    input_text = st.text_input('PDF网址', '')
    input_text1 = st.text_input(':blue[提问]','')
    if st.button('回答'):
        if not qa:
            query = input_text1
    #         result = qa({"query": query})
            prompt3 = CustomPromptTemplate(
            template=template3,
            tools_getter=get_tools,
            input_variables=["input", "intermediate_steps"])
            llm_chain = LLMChain(llm=llm, prompt=prompt3)
            tools = [
                Tool(
                    name = "ZGPA",
                    func=中国平安,
                    description="当您需要回答有关中国平安(601318)财报信息的问题时，这个工具非常有用。"
                ),
                Tool(
                    name = "Google",
                    func=search.run,
                    description="当您需要回答有关当前财经管理问题时，这个工具非常有用。"
                )]
            tool_names = [tool.name for tool in tools]
            agent3 = LLMSingleActionAgent(
                llm_chain=llm_chain, 
                output_parser=output_parser,
                stop=["\nObservation:"], 
                allowed_tools=tool_names
            )
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent3, tools=tools, verbose=True,return_intermediate_steps=True)
            response = agent_executor({"input":query})
            st.write(response["intermediate_steps"])
            st.write(response["output"])
        else:
            query = input_text1
    #         result = qa.run(query)
            tools = [Tool(
                name = "上传",
                func=qa.run,
                description="当您需要回答有关上传公司财报信息的问题时，这个工具非常有用。"
                ),
                      Tool(
                    name = "Google",
                    func=search.run,
                    description="当您需要回答有关当前财经管理问题时，这个工具非常有用。"
                ),
    #                  Tool(
    #                 name="维基",
    #                 func=wikipedia.run,
    #                 description="这个工具适用于当您需要回答有关财经管理问题的名词解释时，输入转换为英文，输出转换为中文"
    #             ),

               ]
    #         result = qa({"query": query})
            tool_names = [tool.name for tool in tools]
            prompt_Upload = CustomPromptTemplate_Upload(
            template=template3,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"])

            llm_chain = LLMChain(llm=llm, prompt=prompt_Upload)
            agent3 = LLMSingleActionAgent(
                llm_chain=llm_chain, 
                output_parser=output_parser,
                stop=["\nObservation:"], 
                allowed_tools=tool_names
            )
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent3, tools=tools, verbose=True,return_intermediate_steps=True)
            response = agent_executor({"input":query})
            st.write(response["intermediate_steps"])
            st.write(response["output"])
    input_text3 = st.text_input(':blue[市场表现提问]','')
    if st.button('回答', key='cninfo财务数据'):
        a=getseccode(input_text3)
        agent_df = cnifo(a)
        response=agent_df({"input":input_text3})
        st.write(response["output"])
        st.json(response["intermediate_steps"])

    # st.header("总结系统")
    # if st.button('总结'):
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         # Set a really small chunk size, just to show.
    #         chunk_size=400,
    #         chunk_overlap=40,
    #         length_function=len,
    #     )
    #     if file is not None:
    #         with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
    #             tmp_file.write(file.read())
    #             tmp_file.flush()
    #             loader = PyPDFLoader(tmp_file.name)
    #             documents = loader.load()
    #             texts = text_splitter.split_documents(documents)
    #             chain = load_summarize_chain(llm=ChatOpenAI(
    #                 model_name="gpt-3.5-turbo",
    #                 temperature=0,
    #                 frequency_penalty=0,
    #                 presence_penalty=0,
    #                 top_p=1.0,
    #             ), chain_type="map_reduce", map_prompt=PROMPT1, combine_prompt=PROMPT1)
    #             st.write(chain.run(texts))
    #     elif input_text != "":        
    #         loader = PyPDFLoader(input_text)
    #         documents = loader.load()
    #         texts = text_splitter.split_documents(documents)
    #         chain = load_summarize_chain(llm=ChatOpenAI(
    #             model_name="gpt-3.5-turbo",
    #             temperature=0,
    #             frequency_penalty=0,
    #             presence_penalty=0,
    #             top_p=1.0,
    #         ), chain_type="map_reduce", map_prompt=PROMPT1, combine_prompt=PROMPT1)
    #         st.write(chain.run(texts))
    #     else:
    #         st.warning("请先加载文档。")
    #     documents = loader.load()


