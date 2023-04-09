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
import tempfile
global bb
bb = ""
file = st.file_uploader("上传PDF文件", type="pdf")
input_text = st.text_input('PDF网址', '')
@st.cache(allow_output_mutation=True)
def 分析(input_text):
    global bb
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
    bb=documents[0].page_content[:100]

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
qa = 分析(input_text)
st.write(bb)

input_text1 = st.text_input('查询', '')
if st.button('查询'):
    if not qa:
        st.warning("请先加载文档。")
    else:
        query = input_text1
        result = qa.run(query)
        st.write(result)



