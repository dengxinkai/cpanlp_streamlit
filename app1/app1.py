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
import math
import pickle
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from collections import deque
from typing import List, Union,Callable,Dict, Optional, Any, Tuple
from langchain.agents import  ZeroShotAgent,AgentExecutor, LLMSingleActionAgent, AgentOutputParser,initialize_agent, Tool,AgentType,create_pandas_dataframe_agent
from langchain.prompts import StringPromptTemplate,PromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.schema import BaseLanguageModel,AgentAction, AgentFinish,Document
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
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
st.set_page_config(
    page_title="数字人",
    page_icon="https://raw.githubusercontent.com/dengxinkai/cpanlp_streamlit/main/app/%E6%9C%AA%E5%91%BD%E5%90%8D.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.cpanlp.com/',
        'Report a bug': "https://www.cpanlp.com/",
        'About': "可读-财报GPT"
    }
)
# 从本地文件中读取会话状态
def load_session_state():
    with open("filenamelst.pickle", "rb") as f:
        return pickle.load(f)
    

# 将会话状态保存到本地文件
def save_session_state(session_state):
    with open("filenamelst.pickle", "wb") as f:
        pickle.dump(session_state, f)
# 将 st.session 对象转换为 JSON 格式的文本数据
def serialize_session(session):
    return json.dumps(session)

# 将 JSON 格式的文本数据转换为 st.session 对象
def deserialize_session(text):
    return json.loads(text)

# 将 st.session 对象持久化到磁盘上
def save_session(session, file_path):
    with open(file_path, 'w') as f:
        f.write(serialize_session(session))

# 从磁盘上加载 st.session 对象
def load_session(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return deserialize_session(text)

# 保存 st.session 对象到文件中

with st.sidebar:
    if 'input_api' in st.session_state:
        st.text_input(st.session_state["input_api"], key="input_api",label_visibility="collapsed")
    else:
        st.info('请先输入正确的openai api-key')
        st.text_input('api-key','', key="input_api")
    with st.expander("ChatOpenAI属性设置"):
        temperature = st.slider("`temperature`", 0.01, 0.99, 0.3,help="用于控制生成文本随机性和多样性的参数。较高的温度值通常适用于生成较为自由流畅的文本，而较低的温度值则适用于生成更加确定性的文本。")
        frequency_penalty = st.slider("`frequency_penalty`", 0.01, 0.99, 0.3,help="用于控制生成文本中单词重复频率的技术。数值越大，模型对单词重复使用的惩罚就越严格，生成文本中出现相同单词的概率就越低；数值越小，生成文本中出现相同单词的概率就越高。")
        presence_penalty = st.slider("`presence_penalty`", 0.01, 0.99, 0.3,help="用于控制语言生成模型生成文本时对输入提示的重视程度的参数。presence_penalty的值较低，模型生成的文本可能与输入提示非常接近，但缺乏创意或原创性。presence_penalty设置为较高的值，模型可能生成更多样化、更具原创性但与输入提示较远的文本。")
        top_p = st.slider("`top_p`", 0.01, 0.99, 0.3,help="用于控制生成文本的多样性，较小的top_p值会让模型选择的词更加确定，生成的文本会更加一致，而较大的top_p值则会让模型选择的词更加多样，生成的文本则更加多样化。")
        model = st.radio("`模型选择`",
                                ("gpt-3.5-turbo",
                                "gpt-4"),
                                index=0)
filenamelst_abspathname = os.path.abspath('filenamelst.pickle')
st.write(filenamelst_abspathname)        
agent_keys = [key for key in st.session_state.keys() if key.startswith('agent')]   
if st.button("从本地文件中恢复会话状态"):
    st.session = load_session('session.json')

    st.write("已从本地文件中恢复会话状态")

# 如果点击了保存按钮，则将会话状态保存到本地文件
if st.button("保存会话状态到本地文件"):
    save_session(st.session, 'session.json')
    st.write("已将会话状态保存到本地文件")
if st.button('刷新页面'):
    st.experimental_rerun()
if agent_keys:
    st.write("当前数字人：")
    for i,key in enumerate(agent_keys):
        y=st.session_state[key]
        col1, col2 = st.columns([5, 1])
        with col1:
            st.write(f"{i+1}、姓名：",y.name,"，特征：",y.traits,"，状态：",y.status)
        with col2:
            if st.button('删除',key=f"del_{key}"):
                del st.session_state[key]
        if st.button('总结',help="总结",key=f"sum_{key}",type="primary"):
            start_time = time.time()
            with get_openai_callback() as cb:
                st.success(st.session_state[key].get_summary(force_refresh=True))
                with st.expander("费用"):
                    st.success(f"Total Tokens: {cb.total_tokens}")
                    st.success(f"Prompt Tokens: {cb.prompt_tokens}")
                    st.success(f"Completion Tokens: {cb.completion_tokens}")
                    st.success(f"Total Cost (USD): ${cb.total_cost}")
            end_time = time.time()
            st.write(f"采访用时：{round(end_time-start_time,2)} 秒")
else:
    st.write("当前不存在数字人") 
tab1, tab2, tab3,tab4 = st.tabs(["数字人创建", "新的观察与记忆", "数字人访问","数字人对话"])
USER_NAME = "Person A" # The name you want to use when interviewing the agent.
LLM = ChatOpenAI(
        model_name=model,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=top_p,
        openai_api_key=st.session_state.input_api
    ) 
agents={}
class GenerativeAgent(BaseModel):
    name: str
    age: int
    traits: str
    status: str
    llm: BaseLanguageModel
    memory_retriever: TimeWeightedVectorStoreRetriever
    verbose: bool = False
    
    reflection_threshold: Optional[float] = None
    
    current_plan: List[str] = []    
    summary: str = ""  #: :meta private:
    summary_refresh_seconds: int= 3600  #: :meta private:
    last_refreshed: datetime =Field(default_factory=datetime.now)  #: :meta private:
    daily_summaries: List[str] #: :meta private:
    memory_importance: float = 0.0 #: :meta private:
    max_tokens_limit: int = 1200 #: :meta private:
    
    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        lines = re.split(r'\n', text.strip())
        return [re.sub(r'^\s*\d+\.\s*', '', line).strip() for line in lines]


    def _compute_agent_summary(self):
        """"""
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the"
            +" following statements:\n"
            +"{related_memories}"
            + "Do not embellish."
            +"\n\nSummary: "
            +"输出用中文"
        )
        # The agent seeks to think about their core characteristics.
        relevant_memories = self.fetch_memories(f"{self.name}'s core characteristics")
        relevant_memories_str = "\n".join([f"{mem.page_content}" for mem in relevant_memories])
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        return chain.run(name=self.name, related_memories=relevant_memories_str).strip()
    
    def _get_topics_of_reflection(self, last_k: int = 50) -> Tuple[str, str, str]:
        """Return the 3 most salient high-level questions about recent observations."""
        prompt = PromptTemplate.from_template(
            "{observations}\n\n"
            + "Given only the information above, what are the 3 most salient"
            + " high-level questions we can answer about the subjects in the statements?"
            + " Provide each question on a new line.\n\n"
                        +"输出用中文"

        )
        reflection_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        observations = self.memory_retriever.memory_stream[-last_k:]
        observation_str = "\n".join([o.page_content for o in observations])
        result = reflection_chain.run(observations=observation_str)
        return self._parse_list(result)
    
    def _get_insights_on_topic(self, topic: str) -> List[str]:
        """Generate 'insights' on a topic of reflection, based on pertinent memories."""
        prompt = PromptTemplate.from_template(
            "Statements about {topic}\n"
            +"{related_statements}\n\n"
            + "What 5 high-level insights can you infer from the above statements?"
            + " (example format: insight (because of 1, 5, 3))"
                        +"输出用中文"

        )
        related_memories = self.fetch_memories(topic)
        related_statements = "\n".join([f"{i+1}. {memory.page_content}" 
                                        for i, memory in 
                                        enumerate(related_memories)])
        reflection_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        result = reflection_chain.run(topic=topic, related_statements=related_statements)
        # TODO: Parse the connections between memories and insights
        return self._parse_list(result)
    
    def pause_to_reflect(self) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        print(colored(f"Character {self.name} is reflecting", "blue"))
        new_insights = []
        topics = self._get_topics_of_reflection()
        for topic in topics:
            insights = self._get_insights_on_topic( topic)
            for insight in insights:
                self.add_memory(insight)
            new_insights.extend(insights)
        return new_insights
    
    def _score_memory_importance(self, memory_content: str, weight: float = 0.15) -> float:
        """Score the absolute importance of the given memory."""
        # A weight of 0.25 makes this less important than it
        # would be otherwise, relative to salience and time
        prompt = PromptTemplate.from_template(
         "On the scale of 1 to 10, where 1 is purely mundane"
         +" (e.g., brushing teeth, making bed) and 10 is"
         + " extremely poignant (e.g., a break up, college"
         + " acceptance), rate the likely poignancy of the"
         + " following piece of memory. Respond with a single integer."
         + "\nMemory: {memory_content}"
         + "\nRating: "
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        score = chain.run(memory_content=memory_content).strip()
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(score[0]) / 10) * weight
        else:
            return 0.0


    def add_memory(self, memory_content: str) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        importance_score = self._score_memory_importance(memory_content)
        self.memory_importance += importance_score
        document = Document(page_content=memory_content, metadata={"importance": importance_score})
        result = self.memory_retriever.add_documents([document])

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (self.reflection_threshold is not None 
            and self.memory_importance > self.reflection_threshold
            and self.status != "Reflecting"):
            old_status = self.status
            self.status = "Reflecting"
            self.pause_to_reflect()
            # Hack to clear the importance from reflection
            self.memory_importance = 0.0
            self.status = old_status
        return result
    
    def fetch_memories(self, observation: str) -> List[Document]:
        return self.memory_retriever.get_relevant_documents(observation)
    def get_summary(self, force_refresh: bool = False) -> str:
        current_time = datetime.now()
        since_refresh = (current_time - self.last_refreshed).seconds
        if not self.summary or since_refresh >= self.summary_refresh_seconds or force_refresh:
            self.summary = self._compute_agent_summary()
            self.last_refreshed = current_time
        return (
            f"姓名: {self.name} (age: {self.age})"
            +f"\n内在特质: {self.traits}"
            +f"\n{self.summary}"
        )
    def get_full_header(self, force_refresh: bool = False) -> str:
        """Return a full header of the agent's status, summary, and current time."""
        summary = self.get_summary(force_refresh=force_refresh)
        current_time_str =  datetime.now().strftime("%B %d, %Y, %I:%M %p")
        return f"{summary}\nIt is {current_time_str}.\n{self.name}'s status: {self.status}"
    def _get_entity_from_observation(self, observation: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the observed entity in the following observation? {observation}"
            +"\nEntity="
            +"输出用中文"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        return chain.run(observation=observation).strip()
    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the {entity} doing in the following observation? {observation}"
            +"\nThe {entity} is"
            +"输出用中文"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        return chain.run(entity=entity_name, observation=observation).strip()
    def _format_memories_to_summarize(self, relevant_memories: List[Document]) -> str:
        content_strs = set()
        content = []
        for mem in relevant_memories:
            if mem.page_content in content_strs:
                continue
            content_strs.add(mem.page_content)
            created_time = mem.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
            content.append(f"- {created_time}: {mem.page_content.strip()}")
        return "\n".join([f"{mem}" for mem in content])
    def summarize_related_memories(self, observation: str) -> str:
        """Summarize memories that are most relevant to an observation."""
        entity_name = self._get_entity_from_observation(observation)
        entity_action = self._get_entity_action(observation, entity_name)
        q1 = f"What is the relationship between {self.name} and {entity_name}"
        relevant_memories = self.fetch_memories(q1) # Fetch memories related to the agent's relationship with the entity
        q2 = f"{entity_name} is {entity_action}"
        relevant_memories += self.fetch_memories(q2) # Fetch things related to the entity-action pair
        context_str = self._format_memories_to_summarize(relevant_memories)
        prompt = PromptTemplate.from_template(
            "{q1}?\nContext from memory:\n{context_str}\nRelevant context: "
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        return chain.run(q1=q1, context_str=context_str.strip()).strip()
    
    def _get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Reduce the number of tokens in the documents."""
        result = []
        for doc in self.memory_retriever.memory_stream[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_tokens_limit:
                result.append(doc.page_content) 
        return "; ".join(result[::-1])
    
    def _generate_reaction(
        self,
        observation: str,
        suffix: str
    ) -> str:
        """React to a given observation."""
        prompt = PromptTemplate.from_template(
                "{agent_summary_description}"
                +"\nIt is {current_time}."
                +"\n{agent_name}'s status: {agent_status}"
                + "\nSummary of relevant context from {agent_name}'s memory:"
                +"\n{relevant_memories}"
                +"\nMost recent observations: {recent_observations}"
                + "\nObservation: {observation}"
                + "\n\n" + suffix
        )
        agent_summary_description = self.get_summary()
        relevant_memories_str = self.summarize_related_memories(observation)
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        kwargs = dict(agent_summary_description=agent_summary_description,
                      current_time=current_time_str,
                      relevant_memories=relevant_memories_str,
                      agent_name=self.name,
                      observation=observation,
                     agent_status=self.status)
        consumed_tokens = self.llm.get_num_tokens(prompt.format(recent_observations="", **kwargs))
        kwargs["recent_observations"] = self._get_memories_until_limit(consumed_tokens)
        action_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        result = action_prediction_chain.run(**kwargs)
        return result.strip()
    
    def generate_reaction(self, observation: str) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            "Should {agent_name} react to the observation, and if so,"
            +" what would be an appropriate reaction? Respond in one line."
            +' If the action is to engage in dialogue, write:\nSAY: "what to say"'
            +"\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
            + "\nEither do nothing, react, or say something but not both.\n\n"
                        +"输出用中文，除了SAY:、REACT:等标志词"

        )
        full_result = self._generate_reaction(observation, call_to_action_template)
        result = full_result.strip().split('\n')[0]
        self.add_memory(f"{self.name} 观察到 {observation} 同时反应了 {result}")
        if "REACT:" in result:
            reaction = result.split("REACT:")[-1].strip()
            return False, f"{reaction}"
        if "SAY:" in result:
            said_value = result.split("SAY:")[-1].strip()
            return True, f"{self.name} 说 {said_value}"
        else:
            return False, result

    def generate_dialogue_response(self, observation: str) -> Tuple[bool, str]:
        call_to_action_template = (
            '{agent_name} 会说什么？结束对话，请写：GOODBYE:"要说的话"。否则，要继续对话，请写：SAY:"接下来要说的话"\n\n'
            # 'What would {agent_name} say? To end the conversation, write: GOODBYE: "what to say". Otherwise to continue the conversation, write: SAY: "what to say next"\n\n'
        )
        full_result = self._generate_reaction(observation, call_to_action_template)
        result = full_result.strip().split('\n')[0]
        if "GOODBYE:" in result:
            farewell = result.split("GOODBYE:")[-1].strip()
            self.add_memory(f"{self.name} 观察到 {observation} 同时说 {farewell}")
            return False, f"{self.name} 说：{farewell}"
        if "SAY:" in result:
            response_text = result.split("SAY:")[-1].strip()
            self.add_memory(f"{self.name} 观察到 {observation} and 同时说 {response_text}")
            return True, f"{self.name} 说：{response_text}"
        else:
            return False, result
def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)

def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
    return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)  
def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{USER_NAME} 说 {message}"
    return agent.generate_dialogue_response(new_message)[1]
def run_conversation(agents: List[GenerativeAgent], initial_observation: str) -> None:
    """Runs a conversation between agents."""
    _, observation = agents[1].generate_reaction(initial_observation)
    print(observation)
    turns = 0
    while True:
        break_dialogue = False
        for agent in agents:
            stay_in_dialogue, observation = agent.generate_dialogue_response(observation)
            st.success(observation)
            # observation = f"{agent.name} said {reaction}"
            if not stay_in_dialogue:
                break_dialogue = True   
        if break_dialogue:
            break
        turns += 1

with tab1:
    global agentss
    name = st.text_input('姓名','邓新凯', key="name_input1_6")
    age = st.number_input('年龄',min_value=0, max_value=100, value=20, step=1, key="name_input1_8")
    traits = st.text_input('特征','既内向也外向，渴望成功', key="name_input1_4",help="性格特征，不同特征用逗号分隔")
    status = st.text_input('状态','博士在读，创业实践中', key="status_input1_5",help="状态，不同状态用逗号分隔")
    reflection_threshold = st.slider("反思阈值",min_value=1, max_value=10, value=8, step=1, key="name_input1_9",help="当记忆的总重要性超过该阈值时，模型将停止反思，即不再深入思考已经记住的内容。设置得太高，模型可能会忽略一些重要的信息；设置得太低，模型可能会花费过多时间在不太重要的信息上，从而影响学习效率。")
    memory = st.text_input('记忆','妈妈很善良，喜欢看动漫', key="mery_input1_5",help="记忆，不同记忆用分号分隔")
    if st.button('创建',help="创建数字人",type="primary"):
        global agent1
        global agentss
        agent1 = GenerativeAgent(name=name, 
          age=age,
          traits=traits,
          status=status,
          memory_retriever=create_new_memory_retriever(),
          llm=LLM,
          daily_summaries = [
               "正在为创业找寻合作伙伴",
               "正在烦恼如何博士毕业",
               "吃饭不规律",
           ],
           reflection_threshold = reflection_threshold, # we will give this a relatively low number to show how reflection works
         )
        memory_list = memory.split(";")
        for memory in memory_list:
            agent1.add_memory(memory)      
        st.session_state[f"agent_{name}"] = agent1
with tab2:   
    if agent_keys:  
        updates = []
        for key in agent_keys:
            updates.append(st.session_state[key].name)
        option = st.selectbox("更新人选择",
        (updates), key="update")
        memory = st.text_input('记忆更新','', key="update_memo",help="新记忆，不同新记忆用分号分隔")
        if st.button('确认',help="记忆更新",type="primary"):
            memory_list = memory.split(";")
            for key in agent_keys:
                if getattr(st.session_state[key], 'name') == option:
                    for memory in memory_list:
                        st.session_state[key].add_memory(memory)   
        observ = st.text_input('观察更新','', key="update_observ",help="新观察，不同新观察用分号分隔")
        if st.button('确认',help="观察更新",type="primary"):
            start_time = time.time()
            observ_list = observ.split(";")
            with get_openai_callback() as cb:
                for key in agent_keys:
                    if getattr(st.session_state[key], 'name') == option:
                        for i, observation in enumerate(observ_list):
                            _, reaction = st.session_state[key].generate_reaction(observation)
                            st.write(f"{i+1}、 {observation}")
                            st.success(reaction)
                        with st.expander("费用"):
                            st.success(f"Total Tokens: {cb.total_tokens}")
                            st.success(f"Prompt Tokens: {cb.prompt_tokens}")
                            st.success(f"Completion Tokens: {cb.completion_tokens}")
                            st.success(f"Total Cost (USD): ${cb.total_cost}")
            end_time = time.time()
            st.write(f"采访用时：{round(end_time-start_time,2)} 秒")
with tab4:
    if len(agent_keys) > 1: 
        diags = []
        for key in agent_keys:
            diags.append(st.session_state[key].name)
        diag1 = st.selectbox(
        "第一对话人选择?",
        (diags), key="diag1")
        diag2 = st.selectbox(
        "第二对话人选择?",
        (diags), key="diag2")
        if diag2 == diag1:
            st.write(diag1,"自问自答道：", key="diagself")
        else:
            st.write(diag1, "询问",diag2, key="diag2")
        diag = st.text_input('', key="diaglogue",label_visibility="collapsed")
        if st.button('对话',help="对话生成",type="primary"):
            start_time = time.time()
            diagagents=[]
            with get_openai_callback() as cb:
                for key in agent_keys:
                    if getattr(st.session_state[key], 'name') == diag1:
                        diagagents.append(st.session_state[key])
                for key in agent_keys:
                    if getattr(st.session_state[key], 'name') == diag2: 
                        diagagents.append(st.session_state[key])
                run_conversation(diagagents, diag)
                with st.expander("费用"):
                    st.success(f"Total Tokens: {cb.total_tokens}")
                    st.success(f"Prompt Tokens: {cb.prompt_tokens}")
                    st.success(f"Completion Tokens: {cb.completion_tokens}")
                    st.success(f"Total Cost (USD): ${cb.total_cost}")
            end_time = time.time()
            st.write(f"采访用时：{round(end_time-start_time,2)} 秒")
with tab3:            
    if agent_keys:  
        interws = []
        for key in agent_keys:
            interws.append(st.session_state[key].name)
        option = st.selectbox(
        "采访人选择?",
        (interws), key="intero")
        interview = st.text_input('采访','你怎么看待', key="interview")
        if st.button('采访',help="采访",type="primary"):
            start_time = time.time()
            with get_openai_callback() as cb:
                for key in agent_keys:
                    if getattr(st.session_state[key], 'name') == option:
                        st.success(interview_agent(st.session_state[key], interview))
                        with st.expander("费用"):
                            st.success(f"Total Tokens: {cb.total_tokens}")
                            st.success(f"Prompt Tokens: {cb.prompt_tokens}")
                            st.success(f"Completion Tokens: {cb.completion_tokens}")
                            st.success(f"Total Cost (USD): ${cb.total_cost}")
            end_time = time.time()
            st.write(f"采访用时：{round(end_time-start_time,2)} 秒")

            

    
