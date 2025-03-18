from langchain_ollama import ChatOllama

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from tools.web_content_retriever_tool import WebContentRetrieverTool

class RAGKnowledgeAgent:
    def __init__(self, model: str, memory: bool, config: dict):
        self.model = ChatOllama(model=model)
        if memory:
            self.memory = MemorySaver()
        self.config = config
        self.webSearchTool = WebContentRetrieverTool()
        self.agent_executor = create_react_agent(model=self.model, tools=[self.webSearchTool], checkpointer=self.memory)
    
    def stream(self, text: str):
        for step in self.agent_executor.stream(
            {"messages": [
                SystemMessage(content="YOU ARE A KNOWLEDGE RETRIEVER AGENT. YOU ARE SUPPOSED TO EXTRACT INFORMATION USING THE TOOLS GIVEN TO YOU (web_content_retriever). ALWAYS PREFER WEBSITES LIKE CONGRESS.GOV OR GOVINFO. SUMMARIZE THE ACTS AND WHAT THEY MEAN IN APPROPRIATE NUMBER OF SENTENCES."),
                HumanMessage(content=f"LAWS AND REGULATION ABOUT [{text}] site: congress.gov")
            ]},
            config=self.config,
            stream_mode="values"
        ):
            step["messages"][-1].pretty_print()

# ##############################################################################
# # USAGE GUIDE
# ##############################################################################
# model_name = "llama3.1"
# config = {"configurable": {"thread_id": "abc123"}}
# agent = RAGKnowledgeAgent(model=model_name, memory=True, config=config)

# text = "unfair or deceptive acts or practices in transactions relating to tokens"

# agent.stream(text)

# ##############################################################################
# # OUTPUT
# ##############################################################################
# python3 agents/knowledge.py 
# ================================ Human Message =================================

# LAWS AND REGULATION ABOUT [unfair or deceptive acts or practices in transactions relating to tokens] site: congress.gov
# ================================== Ai Message ==================================
# Tool Calls:
#   web_content_retriever (ecae4a13-e135-4bc4-85b4-1f2e6a840f75)
#  Call ID: ecae4a13-e135-4bc4-85b4-1f2e6a840f75
#   Args:
#     query: LAWS AND REGULATION ABOUT unfair or deceptive acts or practices in transactions relating to tokens site:congress.gov
# Scraping content from: https://www.congress.gov/crs-product/IF12244
# Stored content from: https://www.congress.gov/crs-product/IF12244 in Qdrant
# Scraping content from: https://www.congress.gov/crs_external_products/IF/PDF/IF12244/IF12244.1.pdf
# Stored content from: https://www.congress.gov/crs_external_products/IF/PDF/IF12244/IF12244.1.pdf in Qdrant
# Scraping content from: https://www.congress.gov/114/plaws/publ258/PLAW-114publ258.pdf
# Some characters could not be decoded, and were replaced with REPLACEMENT CHARACTER.
# Stored content from: https://www.congress.gov/114/plaws/publ258/PLAW-114publ258.pdf in Qdrant

# ================================== Ai Message ==================================

# Based on the text, it appears to be a snippet from the Federal Acquisition Regulation (FAR) related to commercial products and services procurement. The main topics covered in this snippet are:

# 1. Evaluation of Offers: The FAR describes how to evaluate offers for commercial products or services, including selecting the most advantageous offer based on factors contained in the solicitation.
# 2. Combined Synopsis/Solicitation Procedure: This procedure combines the synopsis required by 5.203 and the issuance of the solicitation into a single document. The contracting officer must prepare a synopsis as described at 5.207, including additional information such as the solicitation number, evaluation criteria, and contract terms.

# The snippet provides guidance on how to conduct procurements for commercial products or services, emphasizing the importance of following proper procedures to ensure fair and transparent evaluations.

# To answer the user's question "What are the key points related to commercial products in this text?", I would extract the following key points:

# * Evaluation of Offers: The FAR describes how to evaluate offers for commercial products or services.
# * Combined Synopsis/Solicitation Procedure: This procedure combines the synopsis required by 5.203 and the issuance of the solicitation into a single document.

# These two points capture the main ideas related to commercial products in this text.