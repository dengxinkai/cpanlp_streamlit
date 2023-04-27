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












显示 = ""

st.title('智能财报（中国上市公司）')
