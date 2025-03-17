from langchain_ollama import ChatOllama

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

class SummarizerAgent:
    def __init__(self, model: str, memory: bool, config: dict):
        self.model = ChatOllama(model=model)
        if memory:
            self.memory = MemorySaver()
        self.config = config
        self.agent_executor = create_react_agent(model=self.model, tools=[], checkpointer=self.memory)
    
    def stream(self, text: str):
        for step in self.agent_executor.stream(
            {"messages": [
                SystemMessage(content="YOU ARE AN AGENT WHICH CAN SUMMARIZE LONG TEXT WITHIN 10 TO 15 LINES. YOUR TASK WILL BE TO SUMMARIZE THE TEXT GIVEN TO YOU."),
                HumanMessage(content=text)
            ]},
            config=self.config,
            stream_mode="values"
        ):
            step["messages"][-1].pretty_print()
            
################################################################
# USAGE GUIDE
################################################################
# model_name = "llama3.1"
# config = {"configurable": {"thread_id": "abc123"}}
# agent = SummarizerAgent(model=model_name, memory=True, config=config)

# text = """In the early 1900s, the world saw significant changes, particularly in technology, politics, and culture. 
# Innovations in transportation, such as automobiles and airplanes, revolutionized the way people lived. 
# New political ideologies like communism and fascism emerged, reshaping global power dynamics. 
# The arts also experienced a boom, with movements like modernism challenging traditional notions. 
# World War I had a profound impact on international relations, leading to the creation of new countries and the League of Nations."""

# agent.stream(text)