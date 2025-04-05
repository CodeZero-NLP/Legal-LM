import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextClassifier:
    def __init__(self, model_name: str = "nlpaueb/legal-bigbird-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  

    def classify_document_type(self, text: str,extracted_title: str ) -> str:
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=8192, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()
        predicted_class = self.model.config.id2label.get(predicted_class_idx, "Unknown Category")

        result = {"Predicted Class": predicted_class}

        if extracted_title and extracted_title.lower() != "unknown title":
            result["Extracted Title"] = extracted_title
            #result["Title vs Predicted Match"] = extracted_title.lower() == predicted_class.lower()

        return result

# if __name__ == "__main__":
#     classifier = TextClassifier()
#     document_text = ""
#     result = classifier.classify_document_type(document_text)
#     print("Predicted Document Category:", result)
