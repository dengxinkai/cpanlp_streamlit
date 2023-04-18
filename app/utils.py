template3 = """尽量以少的token准确快速给出Final Answer,You have access to the following tools:
    {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do,尽量少的token
    Action: the action to take, should be one of [{tool_names}],尽量少的token
    Action Input: the input to the action,尽量少的token
    Observation: the result of the action,尽量少的token
    Thought: I now know the final answer,尽量少的token
    Final Answer: the final answer to the original input question,准确快速给出Final Answer
    All inputs 、output and context tokens are totally limited to 3800.
    Question: {input}
    {agent_scratchpad}
    都用中文表示，除了格式中的Question:Thought:Action:Action Input:Observation:Thought:Final Answer
    """
