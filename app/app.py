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
import tempfile
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


st.header("上传系统")
file = st.file_uploader("上传PDF文件", type="pdf")
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

st.header("问答系统")
input_text1 = st.text_input('提问', '')
if st.button('问答'):
    if not qa:
        st.warning("请先加载文档。")
    else:
        query = input_text1
#         result = qa.run(query)
        result = qa({"query": query})

        st.write(result["result"])
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
    

