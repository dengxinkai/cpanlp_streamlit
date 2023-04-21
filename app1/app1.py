import streamlit as st
import wikipedia
import faiss
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
from pydantic import BaseModel, Field
from collections import deque
from typing import List, Union,Callable,Dict, Optional, Any
from langchain.agents import  ZeroShotAgent,AgentExecutor, LLMSingleActionAgent, AgentOutputParser,initialize_agent, Tool,AgentType,create_pandas_dataframe_agent
from langchain.prompts import StringPromptTemplate,PromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.schema import AgentAction, AgentFinish,Document
from langchain.document_loaders import PyPDFLoader
from langchain.docstore import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.llms import OpenAI,BaseLLM
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma,FAISS
from langchain.vectorstores.base import VectorStore
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper,WikipediaAPIWrapper,TextRequestsWrapper
from langchain.callbacks import get_openai_callback
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from utils import template3

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
    st.info('请先输入正确的openai api-key')
    st.text_input('api-key','', key="input_api")
    with st.expander("ChatOpenAI属性设置"):
        temperature = st.slider("`temperature`", 0.01, 0.99, 0.3)
        frequency_penalty = st.slider("`frequency_penalty`", 0.01, 0.99, 0.3,help="在OpenAI GPT语言模型中，温度（temperature）是一个用于控制生成文本随机性和多样性的参数。它可以被理解为对下一个词的概率分布进行重新加权的因子，其中较高的温度值将导致更多的随机性和不确定性，而较低的温度值将导致更少的随机性和更高的确定性。通过调整温度值，可以控制生成文本的风格和多样性，以满足特定的应用需求。较高的温度值通常适用于生成较为自由流畅的文本，而较低的温度值则适用于生成更加确定性的文本。")
        presence_penalty = st.slider("`presence_penalty`", 0.01, 0.99, 0.3)
        top_p = st.slider("`top_p`", 0.01, 0.99, 0.3)
        model = st.radio("`模型选择`",
                                ("gpt-3.5-turbo",
                                "gpt-4"),
                                index=0)
st.title('数字人')
