import streamlit as st
import numpy as np
import pandas as pd
import base64
import os
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
wikipedia = WikipediaAPIWrapper()
llm = OpenAI(temperature=0)

search = GoogleSearchAPIWrapper(google_api_key="AIzaSyCLKh_M6oShQ6rUJiw8UeQ74M39tlCUa9M",google_cse_id="c147e3f22fbdb4316")
global qa
logo_url = "https://raw.githubusercontent.com/dengxinkai/cpanlp_streamlit/main/app/%E6%9C%AA%E5%91%BD%E5%90%8D.png"
st.image(logo_url, width=120)
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
    return RetrievalQA.from_chain_type(llm=ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    frequency_penalty=0,
    presence_penalty=0,
    top_p=1.0,
), chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)
qa = 分析(input_text)

st.header("问答")
input_text1 = st.text_input('提问','')
if st.button('问答'):
    if not qa:
        st.warning("请先加载文档。")
    else:
        query = input_text1
#         result = qa.run(query)
        tools = [
            Tool(
                name = "公司财报",
                func=qa.run,
                description="这个工具适用于当您需要回答有关最近的公司财务报告的问题时。输入应该是一个完整的问题。"
            ),
            Tool(
                name = "查询",
                func=search.run,
                description="这个工具适用于当您需要回答有关当前事件的问题时。"
            ),
            Tool(
                name="wikipedia",
                func=wikipedia.run,
                description="这个工具适用于当您需要回答有关名词解释时。输入应该转换为英文，同时输出转换为中文"
            ),
        ]
#         result = qa({"query": query})
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, return_intermediate_steps=True)
        response = agent({"input":query})
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
    

