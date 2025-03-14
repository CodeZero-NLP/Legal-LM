from langchain_ollama import OllamaLLM

class OllamaClient:
    def __init__(self, model_name: str):
        self.ollama_client = OllamaLLM(model=model_name)
    
    def invoke(self, input):
        return self.ollama_client.invoke(input)