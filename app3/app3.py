import streamlit as st
import asyncio
import faiss
import numpy as np
import pandas as pd
import re
import time
import math
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List, Union,Callable,Dict, Optional, Any, Tuple
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.schema import BaseLanguageModel,Document
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.retrievers import TimeWeightedVectorStoreRetriever
import boto3
st.set_page_config(
    page_title="数字人",
    page_icon="https://raw.githubusercontent.com/dengxinkai/cpanlp_streamlit/main/app1/shuziren.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.cpanlp.com/',
        'Report a bug': "https://www.cpanlp.com/",
        'About': '社科实验数字人'
    }
)
@st.cache_resource
def load_digitalaws():
    dynamodb = boto3.client(
        'dynamodb',
        region_name="ap-southeast-1", 
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    table_name = 'digit_human1'
    response = dynamodb.scan(
        TableName=table_name
    )
    items = response['Items']
    dfaws = pd.DataFrame(items)
    return dfaws
@st.cache_data(persist="disk")
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')
with st.sidebar:
    st.image("https://raw.githubusercontent.com/dengxinkai/cpanlp_streamlit/main/app1/shuziren.jpg")
    with st.expander("👇 :blue[**第一步：输入 OpenAI API 密钥**]"):
        if 'input_api' in st.session_state:
            st.text_input(st.session_state["input_api"], key="input_api",label_visibility="collapsed")
        else:
            st.info('请先输入正确的openai api-key')
            st.text_input('api-key','', key="input_api",type="password")
        temperature = st.slider("`temperature`", 0.01, 0.99, 0.3,help="用于控制生成文本随机性和多样性的参数。较高的温度值通常适用于生成较为自由流畅的文本，而较低的温度值则适用于生成更加确定性的文本。")
        frequency_penalty = st.slider("`frequency_penalty`", 0.01, 0.99, 0.3,help="用于控制生成文本中单词重复频率的技术。数值越大，模型对单词重复使用的惩罚就越严格，生成文本中出现相同单词的概率就越低；数值越小，生成文本中出现相同单词的概率就越高。")
        presence_penalty = st.slider("`presence_penalty`", 0.01, 0.99, 0.3,help="用于控制语言生成模型生成文本时对输入提示的重视程度的参数。presence_penalty的值较低，模型生成的文本可能与输入提示非常接近，但缺乏创意或原创性。presence_penalty设置为较高的值，模型可能生成更多样化、更具原创性但与输入提示较远的文本。")
        top_p = st.slider("`top_p`", 0.01, 0.99, 0.3,help="用于控制生成文本的多样性，较小的top_p值会让模型选择的词更加确定，生成的文本会更加一致，而较大的top_p值则会让模型选择的词更加多样，生成的文本则更加多样化。")
        model = st.radio("`模型选择`",
                                ("gpt-3.5-turbo",
                                "gpt-4"),
                                index=0)
    USER_NAME = st.text_input("请填写创数人姓名","Person", key="user_name")
agent_keys = [key for key in st.session_state.keys() if key.startswith('agent')]   
if st.button('刷新页面'):
    st.experimental_rerun()
    st.cache_data.clear()
if agent_keys:
    do_traits=[]
    with st.expander("当前数字人："):
        for i,key in enumerate(agent_keys):
            y=st.session_state[key]
            col1, col2= st.columns([1, 1])
            with col1:
                do_traits.append(y.traits)
            with col2:
                if st.button('删除',key=f"del_{key}"):
                    del st.session_state[key]
                    st.experimental_rerun()
        df = pd.DataFrame({
                        '特征': do_traits
                    })

        st.dataframe(df, use_container_width=True)
        if st.button('删除所有数字人',key=f"delete_all"):
            for i,key in enumerate(agent_keys):
                del st.session_state[key]
            st.experimental_rerun()
        csv = convert_df(df)
        st.download_button(
           "下载数字人",
           csv,
           "file.csv",
           "text/csv",
           key='download-csv'
        )

else:
    st.warning("当前不存在数字人") 
tab1, tab3 = st.tabs(["数字人创建", ":blue[**社科调查**]"])
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
    traits: str
    llm: BaseLanguageModel
    memory_retriever: TimeWeightedVectorStoreRetriever
    verbose: bool = False
    agent_memory: str= ""
    reflection_threshold: Optional[float] = None
    
    current_plan: List[str] = []    
    summary: str = ""  #: :meta private:
    summary_refresh_seconds: int= 3600  #: :meta private:
    last_refreshed: datetime =Field(default_factory=datetime.now)  #: :meta private:
    daily_summaries: List[str] = [] #: :meta private:
    memory_importance: float = 0.0 #: :meta private:
    max_tokens_limit: int = 1200 #: :meta private:
    class Config:
        arbitrary_types_allowed = True
    @staticmethod
    def _parse_list(text: str) -> List[str]:
        lines = re.split(r'\n', text.strip())
        return [re.sub(r'^\s*\d+\.\s*', '', line).strip() for line in lines]
  
    def _generate_reaction(
        self,
        observation: str,
        suffix: str
    ) -> str:
        """React to a given observation."""
        prompt = PromptTemplate.from_template(
                "{agent_summary_description}"
                +"\nIt is {current_time}."
           +"\n{agent_name} is {traits} and must only give {traits} answers."
                +"\n{agent_name}'s status: {agent_status}"
                + "\nSummary of relevant context from {agent_name}'s memory:"
                +"\n{relevant_memories}"
                +"\nMost recent observations: {recent_observations}"
                + "\nObservation: {observation}"
                + "\n\n" + suffix
                +"输出用中文，除了SAY:、REACT:"

        )
        agent_summary_description = self.get_summary()
        relevant_memories_str = self.summarize_related_memories(observation)
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        kwargs = dict(agent_summary_description=agent_summary_description,
                      current_time=current_time_str,
                      relevant_memories=relevant_memories_str,
                      traits=self.traits,
                      agent_name=self.name,
                      observation=observation,
                     agent_status=self.status)
        consumed_tokens = self.llm.get_num_tokens(prompt.format(recent_observations="", **kwargs))
        kwargs["recent_observations"] = self._get_memories_until_limit(consumed_tokens)
        action_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        result = action_prediction_chain.run(**kwargs)
        return result.strip()
    def generate_reaction(self, observation: str) -> Tuple[bool, str]:
        call_to_action_template = (
            "Should {agent_name} react to the observation, and if so,"
            +" what would be an appropriate reaction? Respond in one line."
            +' If the action is to engage in dialogue, write:\nSAY: "what to say"'
            +"\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
            + "\nEither do nothing, react, or say something but not both.\n\n"
                +"输出用中文，除了SAY:、REACT:"
        )
        full_result = self._generate_reaction(observation, call_to_action_template)
        result = full_result.strip().split('\n')[0]
        self.add_memory(f"{self.name} 观察到 {observation} 同时反应了 {result}")
        if "REACT:" in result or "REACT：" in result or "反应：" in result or "反应:" in result:
            reaction = re.split(r'REACT:|REACT：|反应：|反应:', result)[-1].strip()
            return False, f"{reaction}"
        if "SAY:" in result or "SAY：" in result or "说：" in result or "说:" in result:
            said_value = re.split(r'SAY:|SAY：|说：|说:', result)[-1].strip()
            return True, f"{self.name} 说 {said_value}"
        else:
            return False, result
    def generate_dialogue_response(self, observation: str) -> Tuple[bool, str]:
        call_to_action_template = (
            'What would {agent_name} say? To end the conversation,'
            +'write: GOODBYE: "what to say". '
            +'Otherwise to continue the conversation, write: SAY: "what to say next"\n\n'
            +'输出用中文，除了GOODBYE:、SAY:'
        )
        full_result = self._generate_reaction(observation, call_to_action_template)
        result = full_result.strip().split('\n')[0]
        if "GOODBYE:" in result or "GOODBYE：" in result or "再见：" in result or "再见:" in result:
            farewell = re.split(r'GOODBYE：|GOODBYE:|再见:|再见：', result)[-1].strip()
            self.add_memory(f"{self.name} 观察到 {observation} 同时说 {farewell}")
            self.agent_memory += f"#{self.name} 观察到 {observation} 同时说 {farewell}"
            return False, f"{self.name} 说：{farewell}"
        if "SAY:" in result or "SAY：" in result or "说：" in result or "说:" in result:
            response_text = re.split(r'SAY：|说：|SAY:|说:', result)[-1].strip()
            self.add_memory(f"{self.name} 观察到 {observation} 同时说 {response_text}")
            self.agent_memory += f"#{self.name} 观察到 {observation} 同时说 {response_text}"
            return True, f"{self.name} 说：{response_text}"
        else:
            return False, result
def interview_agent(agent: GenerativeAgent, message: str) -> str:
    new_message = f"{message}"
    return agent.generate_dialogue_response(new_message)[1]
def run_conversation(agents: List[GenerativeAgent], initial_observation: str) -> None:
    _, observation = agents[1].generate_reaction(initial_observation)
    st.success(observation)
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
    with st.expander("单个创建"):
        traits = st.text_input('特征','既内向也外向，渴望成功', key="name_input1_4",help="性格特征，不同特征用逗号分隔")
        if st.button('创建',help="创建数字人"):
            global agent1
            global agentss
            agent1 = GenerativeAgent(
              traits=traits,
              memory_retriever=create_new_memory_retriever(),
              llm=LLM,
              daily_summaries = [
                   "",
               ],
               agent_memory=memory,
               reflection_threshold = reflection_threshold, # we will give this a relatively low number to show how reflection works
             )
            memory_list = re.split(r'#', memory)[1:]
            
            st.session_state[f"agent_{name}"] = agent1
            st.experimental_rerun()
    uploaded_file = st.file_uploader("csv文件上传批量建立", type=["csv"],help="csv格式：姓名、年龄、性别、特征、状态、反思阈值、记忆、总结")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data)
        for index, row in data.iterrows():
            name = row['姓名']
            age = row['年龄']
            gender = row['性别']
            traits = row['特征']
            status = row['状态']
            memory = row['记忆']
            summary = row['总结'] 
            reflection_threshold = row['反思阈值']
            st.session_state[f"agent_{name}"]  = GenerativeAgent(name=name, 
               
                  traits=traits,
                
                  memory_retriever=create_new_memory_retriever(),
                  llm=LLM,
                  daily_summaries = [
                       "",
                   ],
                  agent_memory=memory,
                  summary=summary,
                   reflection_threshold = reflection_threshold, # we will give this a relatively low number to show how reflection works
                 )

with tab3:            
    if agent_keys:
        do_inter_name=[]
        do_inter_quesition=[]
        do_inter_result=[]
        interws = []
        for key in agent_keys:
            interws.append(st.session_state[key].name)
        option = st.selectbox(
        "采访人选择?",
        (interws), key="intero")
        interview = st.text_input('采访','你怎么看待', key="interview")
        if st.button('全部采访',help="全部采访",type="primary",key="quanbu"):
            with st.expander("采访结果",expanded=True):
                start_time = time.time()
                with get_openai_callback() as cb:
                    async def interview_all_agents(agent_keys, interview):
                        tasks = []
                        for key in agent_keys:
                            task = asyncio.create_task(interview_agent_async(st.session_state[key], interview))
                            tasks.append(task)
                        results = await asyncio.gather(*tasks)
                        for key, inter_result in zip(agent_keys, results):
                            st.success(inter_result)
                            do_inter_name.append(st.session_state[key].name)
                            do_inter_quesition.append(interview)
                            do_inter_result.append(inter_result)
                        return do_inter_name,do_inter_quesition, do_inter_result
                    async def interview_agent_async(agent, interview):
                        inter_result = await asyncio.to_thread(interview_agent, agent, interview)
                        return inter_result
                    do_inter_name, do_inter_quesition,do_inter_result = asyncio.run(interview_all_agents(agent_keys, interview))
                    st.success(f"Total Tokens: {cb.total_tokens}")
                    st.success(f"Prompt Tokens: {cb.prompt_tokens}")
                    st.success(f"Completion Tokens: {cb.completion_tokens}")
                    st.success(f"Total Cost (USD): ${cb.total_cost}")
                end_time = time.time()
                st.write(f"采访用时：{round(end_time-start_time,2)} 秒")
        df_inter = pd.DataFrame({
                    '被采访人':do_inter_name,
                    '采访问题':do_inter_quesition,
                    '采访结果': do_inter_result,
                })
        if len(df_inter) > 1:
            question = df_inter.loc[0, '采访问题']
            merged_results = ''.join(df_inter['采访结果'])
            summary_template = """用统计学的方法根据上述回答{answer},对关于{question}问题的回答进行总结，并分析结论是否有显著性?"""
            summary_prompt = PromptTemplate(template=summary_template, input_variables=["answer", "question"])
            llm_chain = LLMChain(prompt=summary_prompt, llm=LLM)
            st.write(llm_chain.predict(answer=merged_results, question=question))
        with st.expander("采访记录"):
            st.dataframe(df_inter, use_container_width=True)
            csv_inter = convert_df(df_inter)
            st.download_button(
               "下载采访记录",
               csv_inter,
               "file.csv",
               "text/csv",
               key='download-csv_inter'
            )
if st.button('aws',key="aws"):

    dfaws = load_digitalaws()
    for index, row in dfaws.iterrows():
        name = row['姓名'].get('S', '')
        age = int(row['年龄'].get('N', ''))
        gender = row['性别'].get('S', '')
        traits = row['特征'].get('S', '')
        status = row['状态'].get('S', '')
        memory = row['记忆'].get('S', '')
        summary = ""
        reflection_threshold = float(row['反思阈值'].get('N', ''))                                  
        st.session_state[f"agent_{name}"]  = GenerativeAgent(name=name, 
              age=age,
              gender=gender,
              traits=traits,
              status=status,
              memory_retriever=create_new_memory_retriever(),
              llm=LLM,
              daily_summaries = [
                   "",
               ],
              agent_memory=memory,
              summary=summary,
               reflection_threshold = reflection_threshold,
             )








