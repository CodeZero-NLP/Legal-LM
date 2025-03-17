from utils.websearcher import WebContentRetriever

from pydantic import BaseModel, Field
from typing import Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema

qdrant_url = "http://localhost:6333"
qdrant_collection_name = "web_content"
num_results = 3
retriever_tool = WebContentRetriever(qdrant_url, qdrant_collection_name, num_results)

class WebContentRetrieverToolInput(BaseModel):
    query: str = Field(description="Input Query to search.")

class WebContentRetrieverTool(BaseTool):
    name: str = "web_content_retriever"
    description: str = "Retrieves and stores web content using DuckDuckGo search and Qdrant for vector storage."
    args_schema: Optional[ArgsSchema] = WebContentRetrieverToolInput

    def _run(self, query: str):
        retriever_tool.store_content_in_qdrant(query)
        return retriever_tool.search_in_qdrant(query)
    
####################################################################
# USAGE GUIDE
####################################################################
# memory = MemorySaver()
# llm = ChatOllama(model="llama3.1")

# web_content_tool = WebContentRetrieverTool()
# tools = [web_content_tool]

# agent_executor = create_react_agent(model=llm, tools=tools, checkpointer=memory)

# config = {"configurable": {"thread_id": "abc123"}}
# for step in agent_executor.stream(
#     {"messages": [HumanMessage(content="Is langchain the best tool in the market ?")]},
#     config,
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()