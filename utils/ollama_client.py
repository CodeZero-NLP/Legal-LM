from langchain_ollama import OllamaLLM
from typing import Dict, Any, Optional, List

class OllamaClient:
    def __init__(self, model_name: str, temperature: float = 0.1, max_tokens: int = 2000):
        """
        Initialize the Ollama client.
        
        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_client = OllamaLLM(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def invoke(self, input):
        """
        Simple invocation with a single input text.
        
        Args:
            input: Input text prompt
            
        Returns:
            str: Generated text response
        """
        return self.ollama_client.invoke(input)
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a completion with separate system and user prompts.
        
        Args:
            system_prompt: System prompt/instructions
            user_prompt: User query/input
            
        Returns:
            str: Generated text response
        """
        # Combine prompts in a format Ollama can understand
        combined_prompt = f"<s>[INST] {system_prompt} [/INST]</s>\n\n[INST] {user_prompt} [/INST]"
        return self.invoke(combined_prompt)
    
    def batch_generate(self, prompts: List[Dict[str, str]]) -> List[str]:
        """
        Generate completions for multiple prompts.
        
        Args:
            prompts: List of dictionaries containing system_prompt and user_prompt
            
        Returns:
            List[str]: List of generated responses
        """
        results = []
        for prompt_dict in prompts:
            response = self.generate(
                prompt_dict.get("system_prompt", ""),
                prompt_dict.get("user_prompt", "")
            )
            results.append(response)
        return results
    