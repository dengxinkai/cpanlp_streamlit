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

result=""
if "prev_input_text" not in st.session_state:
    st.session_state.prev_input_text = ""
st.set_page_config(
    page_title="cpanlp的机器学习",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.cpanlp.com/',
        'Report a bug': "https://www.cpanlp.com/",
        'About': "很高兴您使用cpanlp的机器学习项目"
    }
)
st.write("[返回](https://cpanlp.com/example/)")
input_text = st.text_input('PDF网址', '')
input_text1 = st.text_input('查询', '')

query = input_text1
result = None
do_query = False

import pickle

# 初始化prev_input_text变量
if "prev_input_text" not in st.session_state:
    st.session_state.prev_input_text = ""

# 加载或创建qa对象
if "qa" in st.session_state:
    qa = pickle.loads(st.session_state.qa)
else:
    # 初始化qa对象
    loader = PyPDFLoader(input_text)
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
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

input_text = st.text_input('PDF网址', '')
input_text1 = st.text_input('查询', '')

query = input_text1
result = None
do_query = False

if st.button('确认'):
    if input_text != st.session_state.prev_input_text:
        # 如果input_text变化了，则重新加载文档和QA模型
        loader = PyPDFLoader(input_text)
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
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
        
        # 保存qa对象到session state中
        st.session_state.qa = pickle.dumps(qa)
        
        # 更新prev_input_text的值
        st.session_state.prev_input_text = input_text
        # 执行查询
        result = qa.run(query)
        do_query = True
    else:
        # 如果input_text没有变化，则只执行查询
        result = qa.run(query)
        do_query = True

# 输出结果
if do_query:
    st.write(result)



uploaded_file = st.file_uploader("上传csv文件", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("csv导入成df成功")
    st.write(df)


