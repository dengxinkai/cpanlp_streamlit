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

result = ""
qa = None

input_text = st.text_input('PDF网址', '')

if st.button('确认'):
    try:
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
        global qa
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
    except Exception as e:
        st.error(f"加载文档失败：{e}")
    else:
        st.success("文档加载成功。")

input_text1 = st.text_input('查询', '')
if st.button('查询'):
    global qa
    if not qa:
        st.warning("请先加载文档。")
    else:
        query = input_text1
        result = qa.run(query)
        st.write(result)

uploaded_file = st.file_uploader("上传csv文件", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("csv导入成df成功")
    st.write(df)


