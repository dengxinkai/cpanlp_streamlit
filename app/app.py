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
global qa
import os

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma.db")

input_text = st.text_input('PDF网址', 'http://static.cninfo.com.cn/finalpage/2023-04-08/1216358994.PDF')

@st.cache(allow_output_mutation=True)
def 分析财报(input_text):
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
    return RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
qa = 分析财报(input_text)

input_text1 = st.text_input('查询', '')
if st.button('查询'):
    if not qa:
        st.warning("请先加载文档。")
    else:
        query = input_text1
        result = qa.run(query)
        st.write(result)



