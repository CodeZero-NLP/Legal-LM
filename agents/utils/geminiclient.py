# utils/gemini_client.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  

class GeminiClient:
    def __init__(self, model_name="gemini-pro"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
        response = self.model.generate_content(prompt)
        return response.text.strip() if hasattr(response, 'text') else ""
