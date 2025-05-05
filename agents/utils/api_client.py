# utils/api_client.py
import os
import google.generativeai as genai
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

load_dotenv()

class APIClient:
    """
    A client for Google's Gemini API that follows the same interface pattern
    as the OllamaClient, making it interchangeable in the compliance checker.
    """
    
    def __init__(self, model_name: str = "gemini-pro"):
        """
        Initialize the API client for Google's Gemini model.
        
        Args:
            model_name: The Gemini model to use (e.g., "gemini-pro", "gemini-pro-vision", etc.)
        
        Raises:
            ValueError: If the GOOGLE_API_KEY environment variable is not set
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment. Please set it in .env file.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        # Configure defaults
        self.temperature = 0.1
        self.top_p = 0.95
        self.top_k = 40
        self.max_output_tokens = 4096
        
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Query the Gemini model with a prompt.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to prepend
            
        Returns:
            str: The model's response text
        """
        if system_prompt:
            generation_config = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": self.max_output_tokens,
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=system_prompt
            )
            
            response = model.generate_content(prompt)
        else:
            # Use the default model instance without system prompt
            response = self.model.generate_content(prompt)
        
        return response.text.strip() if hasattr(response, 'text') else ""
    
    def invoke(self, input_text: str) -> str:
        """
        Invoke the model with input text. This method exists for compatibility
        with the OllamaClient interface.
        
        Args:
            input_text: The text to send to the model
            
        Returns:
            str: The model's response text
        """
        return self.query(input_text)
    
    def configure(self, **kwargs) -> None:
        """
        Configure the model parameters.
        
        Args:
            **kwargs: Configuration parameters to set
                - temperature: Controls randomness (0.0 to 1.0)
                - top_p: Controls diversity via nucleus sampling (0.0 to 1.0)
                - top_k: Controls diversity via limiting vocabulary (1 to 100)
                - max_output_tokens: Maximum number of tokens to generate
        """
        if 'temperature' in kwargs:
            self.temperature = kwargs['temperature']
        if 'top_p' in kwargs:
            self.top_p = kwargs['top_p']
        if 'top_k' in kwargs:
            self.top_k = kwargs['top_k']
        if 'max_output_tokens' in kwargs:
            self.max_output_tokens = kwargs['max_output_tokens']