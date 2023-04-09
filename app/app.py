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


template = """给定一个长文档的以下提取部分和一个问题，创建一个带有参考文献（“SOURCES”）的最终答案。
如果您不知道答案，请直接说您不知道。不要试图编造答案。
始终在您的答案中返回“SOURCES”部分。
用意大利语回答。
问题：{question}
最终答案："""

# 定义PromptTemplate对象
prompt_template = PromptTemplate(template=template, input_variables=["question"])

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
    return RetrievalQA.from_chain_type(llm=OpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    frequency_penalty=0,
    presence_penalty=0,
    top_p=1.0,
), chain_type="stuff", retriever=retriever)
qa = 分析(input_text)
st.header("原文前100字")
st.write(bb)
st.header("问答系统")
input_text1 = st.text_input('查询', '')
if st.button('查询'):
    if not qa:
        st.warning("请先加载文档。")
    else:
        query = input_text1
        result = qa.run(query)
        st.write(result)
question = st.text_input("请输入问题", '')
if question:
    prompt = prompt_template(question=question)
    st.markdown(prompt)
    result = 分析(bb + " " + question)
    if result:
        st.write(result.response)
        st.write("SOURCES:", result.sources)
    else:
        st.write("抱歉，无法回答您的问题。")
else:
    st.write("请输入问题以获取答案。")


