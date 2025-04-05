# preprocessor_agent.py
from utils.document_parser import DocumentParser
from utils.roberta_classifier import TextClassifier
from utils.ner import NERModel
from utils.clause_extractor import ClauseExtractor

class PreprocessorAgent:
    def __init__(self):
        self.document_parser = DocumentParser()
        self.text_classifier = TextClassifier()
        self.ner_agent = NERModel()
        self.clause_extractor = ClauseExtractor()

    def process_document(self, file_path: str):
        title, text, *_ = self.document_parser.parse_pdf(file_path)
        clauses = self.clause_extractor.extract_clauses(text)
        document_class = self.text_classifier.classify_document_type(text, title)

        clause_classes = self.text_classifier.classify_clauses(clauses)
        entities = self.ner_agent.extract_entities(text)

        return {
            "Text Extracted" : text,
            "Document Title": title,
            "Document Class": document_class,
            "Important Clauses": "Extracted Clauses": {clause: clause_classes[i] for i, clause in enumerate(clauses)},
            "Named Entities": entities
        }

if __name__ == "__main__":
    agent = PreprocessorAgent()
    file_path = "C:\\Users\\athir\\Downloads\\LegalDoc-1.pdf"  
    result = agent.process_document(file_path)
    
    # print("Text Extracted:", result["Text Extracted"])
    print("Document Title:", result["Document Title"])
    print("Document Class:", result["Document Class"])
    print("Important Clauses:", result["Important Clauses"])
    print("Named Entities:", result["Named Entities"])




# agents/preprocessor.py
# from typing import Dict, List, Any, Optional
# import os
# import uuid
# from context_bank import ContextBank
# from utils.ollama_client import OllamaClient
# from utils.api_client import APIClient
# from utils.document_parser import DocumentParser
# from utils.ner_extractor import NERExtractor
# from utils.clause_extractor import ClauseExtractor
# from utils.text_classifier import TextClassifier

# class PreprocessorAgent:
#     """
#     Preprocessor Agent that handles document parsing, entity recognition,
#     clause extraction, and classification of legal documents.
#     """
    
#     def __init__(self, use_ollama: bool, model_name: str, context_bank: ContextBank):
#         """
#         Initialize the Preprocessor Agent with its sub-components.
        
#         Args:
#             use_ollama: Whether to use Ollama for local model inference
#             model_name: Name of the model to use
#             context_bank: Shared context bank for all agents
#         """
#         self.context_bank = context_bank
        
#         # Initialize the appropriate client based on configuration
#         if use_ollama:
#             self.llm_client = OllamaClient(model_name)
#         else:
#             self.llm_client = APIClient(model_name)
        
#         # Initialize sub-components
#         self.document_parser = DocumentParser()
#         self.ner_extractor = NERExtractor()
#         self.clause_extractor = ClauseExtractor()
#         self.text_classifier = TextClassifier(use_ollama, model_name)
    
#     def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Process a document through parsing, entity recognition, and clause extraction.
        
#         Args:
#             state: Current state of the workflow
            
#         Returns:
#             Dict: Updated state with preprocessing results
#         """
#         document_path = state.get("document_path")
#         document_id = state.get("document_id")
        
#         # Parse the document
#         document_content, document_metadata = self._parse_document(document_path)
        
#         # Store the document in the context bank
#         if document_content:
#             self.context_bank.add_document(document_id, document_content, document_metadata)
            
#             # Extract named entities
#             entities = self._extract_entities(document_content)
#             if entities:
#                 self.context_bank.add_entities(document_id, entities)
            
#             # Extract and classify clauses
#             clauses = self._extract_clauses(document_content)
#             if clauses:
#                 self.context_bank.add_clauses(document_id, clauses)
            
#             # Update the state with preprocessing results
#             state["preprocessing_complete"] = True
#             state["document_type"] = document_metadata.get("document_type", "Unknown")
#             state["entity_count"] = len(entities)
#             state["clause_count"] = len(clauses)
#             state["next_step"] = "knowledge"  # Next, gather relevant legal knowledge
#         else:
#             # If parsing failed
#             state["preprocessing_complete"] = False
#             state["error"] = "Failed to parse document"
#             state["next_step"] = "complete"  # End the workflow with an error
        
#         return state
    
#     def _parse_document(self, document_path: str) -> tuple[str, Dict[str, Any]]:
#         """
#         Parse a document from its file path.
        
#         Args:
#             document_path: Path to the document file
            
#         Returns:
#             tuple: (document_content, document_metadata)
#         """
#         # Determine file type from extension
#         _, file_extension = os.path.splitext(document_path)
#         file_extension = file_extension.lower()
        
#         document_content = ""
#         document_metadata = {
#             "file_path": document_path,
#             "file_type": file_extension[1:] if file_extension else "unknown"
#         }
        
#         try:
#             # Parse based on file type
#             if file_extension in ['.pdf']:
#                 document_content, metadata = self.document_parser.parse_pdf(document_path)
#                 document_metadata.update(metadata)
#             elif file_extension in ['.docx', '.doc']:
#                 document_content, metadata = self.document_parser.parse_word(document_path)
#                 document_metadata.update(metadata)
#             elif file_extension in ['.txt']:
#                 with open(document_path, 'r', encoding='utf-8') as file:
#                     document_content = file.read()
#             else:
#                 # Unsupported file type
#                 document_metadata["error"] = f"Unsupported file type: {file_extension}"
#                 return "", document_metadata
            
#             # Classify the document type using LLM
#             document_type = self.text_classifier.classify_document_type(document_content)
#             document_metadata["document_type"] = document_type
            
#             return document_content, document_metadata
        
#         except Exception as e:
#             document_metadata["error"] = str(e)
#             return "", document_metadata
    
#     def _extract_entities(self, document_content: str) -> List[Dict[str, Any]]:
#         """
#         Extract named entities from document content.
        
#         Args:
#             document_content: Text content of the document
            
#         Returns:
#             List: Extracted entities with metadata
#         """
#         # Use NER extractor to identify legal entities
#         entities = self.ner_extractor.extract(document_content)
        
#         # Format entities with unique IDs
#         formatted_entities = []
#         for entity in entities:
#             formatted_entities.append({
#                 "id": str(uuid.uuid4()),
#                 "text": entity["text"],
#                 "type": entity["type"],
#                 "start_pos": entity["start_pos"],
#                 "end_pos": entity["end_pos"],
#                 "confidence": entity["confidence"]
#             })
        
#         return formatted_entities
    
#     def _extract_clauses(self, document_content: str) -> List[Dict[str, Any]]:
#         """
#         Extract and classify clauses from document content.
        
#         Args:
#             document_content: Text content of the document
            
#         Returns:
#             List: Extracted clauses with classification
#         """
#         # Extract clauses
#         raw_clauses = self.clause_extractor.extract(document_content)
        
#         # Classify each clause
#         classified_clauses = []
#         for clause in raw_clauses:
#             clause_type = self.text_classifier.classify_clause_type(clause["text"])
#             classified_clauses.append({
#                 "id": str(uuid.uuid4()),
#                 "text": clause["text"],
#                 "type": clause_type,
#                 "start_pos": clause["start_pos"],
#                 "end_pos": clause["end_pos"],
#                 "section": clause.get("section", ""),
#                 "subsection": clause.get("subsection", "")
#             })
        
#         return classified_clauses