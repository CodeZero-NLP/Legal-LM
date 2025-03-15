import spacy
class BlackstoneNER:
    def __init__(self):
        self.nlp = self._load_blackstone()
    
    def _load_blackstone(self):
            return spacy.load("en_blackstone_proto")
        
    def extract_entities(self, text):
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
