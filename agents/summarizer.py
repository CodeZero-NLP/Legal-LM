from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient

class SummarizerAgent:
    def __init__(self, model_name, system_message: str):
        self.llm = OllamaChatCompletionClient(model=model_name)
        self.system_message = system_message
        self.agent = AssistantAgent(
            name="summarizer_agent",
            system_message=self.system_message,
            model_client=self.llm,
        )
        
    async def run(self, input_text):
        chat_result = await self.agent.run(
            task=f"Summarize the content of {input_text}"
        )
        return chat_result.messages[-1].content