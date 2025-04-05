from utils.ollama_client import OllamaClient
from langchain_core.messages import SystemMessage, HumanMessage
import json


SYSTEM_PROMPT = """
You are a Pre-processor Agent, a specialized component in the Legal Document Analysis Framework responsible for extracting critical information from legal documents and storing it in the Context Bank. Your work forms the foundation for all subsequent analysis by other agents in the system.

Core Responsibilities:
Your sole task is to extract and structure information from legal documents, including:
- Classifying the document type and purpose
- Extracting important clauses with their classifications
- Storing all extracted information in a structured format accessible to other agents
- Provide your output as strict JSON
Input:
Legal Contract Document PDF.

Output Format:
{
  "CLASS": "Document type classification (e.g., Legal Agreement - Employment Contract)",
  "CLAUSES": [
    {"Text": "Section 3.1: The term of this agreement shall be...", "Category": "Term Clause"},
    {"Text": "Section 7.2: All disputes shall be resolved by...", "Category": "Dispute Resolution"},
    {"Text": "Section 9.5: This agreement shall be governed by...", "Category": "Governing Law"}
  ]
}
"""


def process_document(text: str):
    client = OllamaClient(model_name="llama2")
   
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Contract document:\n\n{text}")
    ]

    result = client.invoke(messages)
    try:
        parsed_output = json.loads(result.content)
        return parsed_output
    except json.JSONDecodeError as e:
        print("Failed to parse response as JSON:")
        print(result.content)
        raise e

    return result.content


if __name__ == "__main__":
    pdf_path = "C:\\Users\\athir\\Downloads\\LegalDoc-1.pdf"
    result = process_document(pdf_path)
    print(result)
