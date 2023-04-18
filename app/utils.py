
template3 = """尽量以最少的iterations准确快速给出Final Answer,You have access to the following tools:
    {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do,同时准确快速给出Final Answer
    Action: the action to take, should be one of [{tool_names}],同时准确快速给出Final Answer
    Action Input: the input to the action,同时准确快速给出Final Answer
    Observation: the result of the action,同时准确快速给出Final Answer
    Thought: I now know the final answer,同时准确快速给出Final Answer
    Final Answer: the final answer to the original input question,准确快速给出Final Answer
    All inputs and output tokens are limited to 3800.
    Question: {input}
    {agent_scratchpad}
    都用中文表示，除了格式中的Question:Thought:Action:Action Input:Observation:Thought:Final Answer:,总共处理时间不要超过15秒，总的context token不超过2000token。
    be careful,model's maximum context length is 4000 tokens. messages must result in less than 4000 tokens
    """
