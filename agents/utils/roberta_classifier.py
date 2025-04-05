from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class TextClassifier:
    def __init__(self):
        token = ""
        self.tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
        self.model = AutoModelForSequenceClassification.from_pretrained('nlpaueb/legal-bert-base-uncased')        

        print("Label mappings:", self.model.config.id2label)

    def classify_text(self, text: str) -> str:
        """Classifies a given text (either document or clause)."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_label = torch.argmax(outputs.logits, dim=1).item()
        return self.model.config.id2label.get(predicted_label, "Unknown Category")
    
    def classify_clauses(self, clauses: list) -> dict:
        """Classifies each extracted clause separately."""
        return {clause: self.classify_text(clause) for clause in clauses}
    
    # def classify_document(self, text: str) -> str:
    #     max_length = self.tokenizer.model_max_length  # Typically 512 for RoBERTa
    #     tokens = self.tokenizer(text, truncation=False, padding=True, return_tensors="pt")

    #     if tokens.input_ids.shape[1] > max_length:
    #         chunk_size = max_length - 10  
    #         chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
            
    #         labels = [self.classify_text(chunk) for chunk in chunks]
    #         final_predicted_class = max(set(labels), key=labels.count)  # Majority vote
    #     else:
    #         final_predicted_class = self.classify_text(text)

    #     return final_predicted_class

    


        
            
           

