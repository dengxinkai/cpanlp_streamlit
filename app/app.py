import streamlit as st
import numpy as np
import pandas as pd
import base64
import json
import os
from typing import List, Union
from langchain.agents import  AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.schema import AgentAction, AgentFinish
import re

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import tempfile
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.prompts import StringPromptTemplate


wikipedia = WikipediaAPIWrapper()
llm=ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    frequency_penalty=0,
    presence_penalty=0,
    top_p=1.0,
)
search = GoogleSearchAPIWrapper(google_api_key="AIzaSyCLKh_M6oShQ6rUJiw8UeQ74M39tlCUa9M",google_cse_id="c147e3f22fbdb4316")
global qa
logo_url = "https://raw.githubusercontent.com/dengxinkai/cpanlp_streamlit/main/app/%E6%9C%AA%E5%91%BD%E5%90%8D.png"
st.image(logo_url, width=40)
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Chinese:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}
prompt_template1 = """Write a concise summary of the following:


{text}


CONCISE SUMMARY IN CHINESE:"""
PROMPT1 = PromptTemplate(template=prompt_template1, input_variables=["text"])
result = ""
template3 = """Answer the following questions as best you can.尽量把问题和经济金融问题联系在一起， You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do,keep the Thought as concise as possible.
Action: the action to take, should be one of [{tool_names}],keep the Action as concise as possible.
Action Input: the input to the action,keep the Action Input as concise as possible.
Observation: the result of the action,keep the Observation as concise as possible.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer,keep the Thought as concise as possible.
Final Answer: the final answer to the original input question

Begin! Remember to speak in Chinese.

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
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
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()



st.header("上传")
file = st.file_uploader("PDF文件", type="pdf")
input_text = st.text_input('PDF网址', '')
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
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)
qa = 分析(input_text)

st.header("问答")
input_text1 = st.text_input('提问','')
if st.button('问答'):
    if not qa:
        st.warning("请先加载文档。")
        tools = [
            Tool(
                name = "GOOGLE查询",
                func=search.run,
                description="这个工具适用于当您需要回答有关当前金融经济事件的问题时。"
            ),
            Tool(
                name="wikipedia查询",
                func=wikipedia.run,
                description="这个工具适用于当您需要回答与当前问题有关的经济和金融理论时。输入应该转换为英文，同时输出转换为中文"
            )]
#         result = qa({"query": query})
        tool_names = [tool.name for tool in tools]
        prompt3 = CustomPromptTemplate(
        template=template3,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"])

        llm_chain = LLMChain(llm=llm, prompt=prompt3)
        agent3 = LLMSingleActionAgent(
            llm_chain=llm_chain, 
            output_parser=output_parser,
            stop=["\nObservation:"], 
            allowed_tools=tool_names
        )
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent3, tools=tools, verbose=True)
        response = agent_executor({"input":query})
        st.write(response["intermediate_steps"])
        st.write(response["output"])
    else:
        query = input_text1
#         result = qa.run(query)
        tools = [Tool(
            name = "公司财报",
            func=qa.run,
            description="这个工具适用于当您需要回答有关上传的公司财务报告的问题时。输入应该是一个完整的问题。"
            ),
            Tool(
                name = "GOOGLE查询",
                func=search.run,
                description="这个工具适用于当您需要回答有关当前金融经济事件的问题时。"
            ),
            Tool(
                name="wikipedia查询",
                func=wikipedia.run,
                description="这个工具适用于当您需要回答与当前问题有关的经济和金融理论时。输入应该转换为英文，同时输出转换为中文"
            )]
#         result = qa({"query": query})
        tool_names = [tool.name for tool in tools]
        prompt3 = CustomPromptTemplate(
        template=template3,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"])

        llm_chain = LLMChain(llm=llm, prompt=prompt3)
        agent3 = LLMSingleActionAgent(
            llm_chain=llm_chain, 
            output_parser=output_parser,
            stop=["\nObservation:"], 
            allowed_tools=tool_names
        )
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent3, tools=tools, verbose=True)
        response = agent_executor({"input":query})
        st.write(response["intermediate_steps"])
        st.write(response["output"])

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
    

