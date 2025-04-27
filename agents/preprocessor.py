# preprocessor_agent.py
# from utils.document_parser import DocumentParser
from utils.roberta_classifier import TextClassifier
import utils.system_prompt as system_prompt
from utils.clause_extractor import ClauseExtractor
import uuid
import PyPDF2
import re
import spacy


class PreprocessorAgent:
    def __init__(self):
        # self.document_parser = DocumentParser()
        self.text_classifier = TextClassifier()
        self.clause_extractor = ClauseExtractor()

    def process_document(self, file_path: str):
        document_id = str(uuid.uuid4())
        
        # --- Start: Inline parse_pdf logic ---
        text = ""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()

        # inline title‐extraction (formerly extract_title)
        lines = text.split("\n")
        candidates = []
        for i, line in enumerate(lines[:10]):
            clean_line = line.strip()
            if not clean_line or len(clean_line) < 5:
                continue
            score = 0
            if re.match(r"^(CONTRACT|AGREEMENT|PETITION|NOTICE|ORDER|BILL|ACT|STATUTE)\b",
                        clean_line, re.IGNORECASE):
                score += 5
            if re.match(r"^[A-Z\s\-]{5,}$", clean_line):
                score += 2
            if "**" in clean_line or clean_line.center(80) == clean_line:
                score += 1
            candidates.append((clean_line, score))
        title = candidates[0][0] if candidates else "Unknown Title"
        # --- End: Inline parse_pdf logic ---

        llm_output = system_prompt.process_document(text)

        document_class = llm_output.get("CLASS", "")
        clause_classes = llm_output.get("CLAUSES", [])

        
        # --- Start: Inline extract_entities logic ---
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        entities_list = []
        for ent in doc.ents:
            entities_list.append((ent.text, ent.label_))
        entities = entities_list 
        # --- End: Inline extract_entities logic ---

        # Assuming context_bank is initialized elsewhere or passed to __init__
        # self.context_bank.add_document(document_id, text, {
        #     "title": title,
        #     "document_type": document_class,
        #     "source_file": file_path
        # })

        # self.context_bank.add_entities(document_id, entities)
        # self.context_bank.add_clauses(document_id, clause_classes)


        return {
            "Text Extracted" : text,
            "Document Title": title,
            "Document Class": document_class,
            "Important Clauses": {clause["Text"]: clause["Category"] for clause in clause_classes},
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