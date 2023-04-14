import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent,AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
import os
from collections import deque
from typing import Dict, List, Optional, Any
import pinecone 
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.utilities import WikipediaAPIWrapper
from langchain.schema import Document
global 显示
显示 = "你好\n朋友"

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
st.write("[返回](https://cpanlp.com/)")
# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
import faiss
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
st.title("""可读财报-BabyAGI""")

# st.sidebar.header('User Input Parameters')
def 中国平安(input_text):
    pinecone.init(api_key="bd20d2c3-f100-4d24-954b-c17928d1c2da",  # find at app.pinecone.io
                      environment="us-east4-gcp",  # next to api key in console
                      namespace="ZGPA_601318")
    index = pinecone.Index(index_name="kedu")
    a=embeddings.embed_query(input_text)
    www=index.query(vector=a, top_k=1, namespace='ZGPA_601318', include_metadata=True)
    c = [x["metadata"]["text"] for x in www["matches"]]
    return c
embeddings = OpenAIEmbeddings()
llm=ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.5,
    frequency_penalty=1,
    presence_penalty=1,
    top_p=0.5,
)
search = GoogleSearchAPIWrapper(google_api_key="AIzaSyCLKh_M6oShQ6rUJiw8UeQ74M39tlCUa9M",google_cse_id="c147e3f22fbdb4316")

class TaskCreationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create 3 new main tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
            " Be careful,This model's maximum context length is 4097 tokens."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=["result", "task_description", "incomplete_tasks", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
class TaskPrioritizationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " 1"
            " 2"
            " Start the task list with number {next_task_id}."
            " Be careful,This model's maximum context length is 4097 tokens."

        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
todo_prompt = PromptTemplate.from_template("Come up with a todo list of 3 most important items for this objective: {objective}.")
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
tools = [
    Tool(
        name = "ZGPA",
        func=中国平安,
        description="This tool is very useful when you need to answer questions about information of Ping An (601318) in China."
        ),
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name = "TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list of three most important items for that objective. Please be very clear what the objective is!"
    )
]
prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}
All inputs and output tokens are limited to 3800.最后把输出的Final Answer结果翻译成中文
"""
prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["objective", "task", "context","agent_scratchpad"]
)
def get_next_task(task_creation_chain: LLMChain, result: Dict, task_description: str, task_list: List[str], objective: str) -> List[Dict]:
    """Get the next task."""
    incomplete_tasks = ", ".join(task_list)
    response = task_creation_chain.run(result=result, task_description=task_description, incomplete_tasks=incomplete_tasks, objective=objective)
    new_tasks = response.split('\n')
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]
def prioritize_tasks(task_prioritization_chain: LLMChain, this_task_id: int, task_list: List[Dict], objective: str) -> List[Dict]:
    """Prioritize tasks."""
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    response = task_prioritization_chain.run(task_names=task_names, next_task_id=next_task_id, objective=objective)
    new_tasks = response.split('\n')
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    return prioritized_task_list
def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return []
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    return [str(item.metadata['task']) for item in sorted_results]

def execute_task(vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5) -> str:
    """Execute a task."""
    context = _get_top_tasks(vectorstore, query=objective, k=k)
    return execution_chain.run(objective=objective, context=context, task=task)
class BabyAGI(Chain, BaseModel):
    """Controller model for the BabyAGI agent."""

    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: AgentExecutor = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None
        
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\n*****TASK LIST*****\n")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\n*****NEXT TASK*****\n")
        print(str(task["task_id"]) + ": " + task["task_name"])
    def print_task_result(self, result: str):
        global 显示
        print("\n*****TASK RESULT*****\n")
        print(result)
        显示+=result

        
    @property
    def input_keys(self) -> List[str]:
        return ["objective"]
    
    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs['objective']
        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0
        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = execute_task(
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)

                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = get_next_task(
                    self.task_creation_chain, result, task["task_name"], [t["task_name"] for t in self.task_list], objective
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain, this_task_id, list(self.task_list), objective
                    )
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print("\n*****TASK ENDING*****\n")
                break
        return {}

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        verbose: bool = False,
        **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(
            llm, verbose=verbose
        )
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
#         agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent = LLMSingleActionAgent(llm_chain=llm_chain, allowed_tools=tool_names)


        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=agent_executor,
            vectorstore=vectorstore,
            **kwargs
        )
OBJECTIVE = st.text_input('提问','')

llm=ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.5,
    frequency_penalty=1,
    presence_penalty=1,
    top_p=0.5,
)# Logging of LLMChains
verbose=True
# If None, will keep on going forever
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    verbose=verbose,
    max_iterations=max_iterations
)
if st.button('问答'):
    baby_agi({"objective": OBJECTIVE})
text_list = 显示.split('\n')
for i in text_list:
    st.write(i)
