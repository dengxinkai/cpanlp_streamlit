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
import time
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
    st.info('欢迎使用我们的服务！为了确保您的应用程序顺利运行，我们建议您在使用之前输入正确的OpenAI的API密钥。如果您还没有API密钥，请访问OpenAI官方网站以获取更多信息。感谢您选择我们的服务！')
    
st.title('智能财报（中国上市公司）')
if st.session_state.input_api:
    @st.cache(allow_output_mutation=True)
    def 中国平安年报查询(input_text):
        pinecone.init(api_key="bd20d2c3-f100-4d24-954b-c17928d1c2da",  # find at app.pinecone.io
                          environment="us-east4-gcp",  # next to api key in console
                          namespace="ZGPA_601318")
        index = pinecone.Index(index_name="kedu")
        a=embeddings.embed_query(input_text)
        www=index.query(vector=a, top_k=1, namespace='ZGPA_601318', include_metadata=True)
        c = [x["metadata"]["text"] for x in www["matches"]]
        return c
    @st.cache(allow_output_mutation=True)
    def 双汇发展年报查询(input_text):
        namespace="ShHFZ_000895"
        pinecone.init(api_key="bd20d2c3-f100-4d24-954b-c17928d1c2da",  # find at app.pinecone.io
                          environment="us-east4-gcp",  # next to api key in console
                          namespace=namespace)
        index = pinecone.Index(index_name="kedu")
        a=embeddings.embed_query(input_text)
        www=index.query(vector=a, top_k=1, namespace=namespace, include_metadata=True)
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
                    description="当您需要搜索互联网时，这个工具非常有用。"
                )
    zgpa_tool =  Tool(
                    name = "ZGPA",
                    func=中国平安年报查询,
                    description="当您需要回答有关中国平安(601318)问题时，这个工具非常有用。"
                )
    shhfz_tool =  Tool(
                    name = "ShHFZ",
                    func=双汇发展年报查询,
                    description="当您需要回答有关双汇发展(000895)问题时，这个工具非常有用。"
                )
    ALL_TOOLS = [search_tool,zgpa_tool,shhfz_tool]
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
    template3 = """Answer the following questions as best you can.You have access to the following tools:
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
    都用中文表示，除了格式中的Question:Thought:Action:Action Input:Observation:Thought:Final Answer:,总共处理时间不要超过15秒，总的context token不超过2000token。
    be careful,model's maximum context length is 4000 tokens. messages must result in less than 4000 tokens
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
    st.info('欢迎使用我们的分析工具！您可以选择直接使用或是上传财报以进行更加深入的分析。无论您选择哪种方式，我们都将为您提供最好的服务。')
    with st.expander("上传"):
        file = st.file_uploader("PDF文件", type="pdf")
        input_text = st.text_input('PDF网址', '')
        qa = 分析(input_text)
    input_text1 = st.text_input(':blue[查询]','')
    if st.button('确认'):
        start_time = time.time()
        if not qa:
            query = input_text1
            prompt3 = CustomPromptTemplate(
            template=template3,
            tools_getter=get_tools,
            input_variables=["input", "intermediate_steps"])
            llm_chain = LLMChain(llm=llm, prompt=prompt3)
            tools = [
                Tool(
                    name = "ZGPA",
                    func=中国平安年报查询,
                    description="当您需要回答有关中国平安(601318)问题时，这个工具非常有用。"
                ),
                Tool(
                    name = "Google",
                    func=search.run,
                    description="当您需要搜索互联网时，这个工具非常有用。"
                ),
                Tool(
                name = "ShHFZ",
                func=双汇发展年报查询,
                description="当您需要回答有关双汇发展(000895)问题时，这个工具非常有用。"
            )]
            tool_names = [tool.name for tool in tools]
            agent3 = LLMSingleActionAgent(
                llm_chain=llm_chain, 
                output_parser=output_parser,
                stop=["\nAction Input:"], 
                allowed_tools=tool_names
            )
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent3, tools=tools, verbose=True,return_intermediate_steps=True,max_iterations=8)
            response = agent_executor({"input":query})
            st.caption(response["output"])
            with st.expander("查看过程"):
                st.write(response["intermediate_steps"])
        else:
            query = input_text1
            tools = [Tool(
                name = "上传",
                func=qa.run,
                description="当您需要回答有关上传公司财报信息的问题时，这个工具非常有用。"
                ),
                      Tool(
                    name = "Google",
                    func=search.run,
                    description="当您需要搜索互联网时，这个工具非常有用。"
                )
               ]
            tool_names = [tool.name for tool in tools]
            prompt_Upload = CustomPromptTemplate_Upload(
            template=template3,
            tools=tools,
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
            st.caption(response["output"])
            with st.expander("查看过程"):
                st.write(response["intermediate_steps"])
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"项目完成所需时间: {elapsed_time:.2f} 秒")  
