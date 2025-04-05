import spacy

class NERModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))

        return entities
    

def main():
    text = """ARTICLE VIII: MISCELLANEOUS
MISCELLANEOUS
8.01 NOTICES. All notices required hereunder shall be in writing and delivered in person, by e-mail, facsimile, Federal Express, UPS, or by certified or registered mail, return receipt requested, postage prepaid. Such notices shall be addressed as follows:
All notices shall be deemed given when delivered in person or upon depositing said notice in the United States mail with proper postage affixed thereto.
8.02 NON-EXCLUSIVITY. Affiliate's rights to locate Contractors hereunder shall not be exclusive. It is expressly contemplated and understood that Network 1 will utilize other persons and companies to locate Contractors.
8.03 AMENDMENT. Except as otherwise provided herein, this Agreement and the Schedules hereto may not be amended, altered or modified except in writing executed by all parties hereto.
8.04 BENEFITS AND ASSIGNMENTS. This agreement may be assigned or delegated, in whole or in part, by NETWORK 1 without the prior written consent of the other party herein. This agreement may not be assigned or delegated by Affiliate without prior written consent from Network 1. Such consent shall not be unreasonably withheld.
8.05 GOVERNING LAW. All disputes or claims by Payment Data Systems hereunder shall be resolved by arbitration in McLean, Virginia, pursuant to the rules of the American Arbitration Association.
8.06 ARBITRATION. All disputes or claims hereunder shall be resolved by arbitration in McLean, Virginia, pursuant to the rules of the American Arbitration Association.
8.07 SEVERABILITY. The illegality, invalidity or unenforceability of any provision of this Agreement shall not affect the remainder of this Agreement.
8.08 ENTIRE AGREEMENT. This Agreement and the attached Schedules, Exhibits and Addendums hereto contain the entire understanding of the parties hereto and supersede all prior agreements with respect to the subject of this Agreement.

EXECUTED this ________ day of ______________________, in the year ____________.
NETWORK 1 FINANCIAL CORPORATION AFFILIATE
By: ______________________________________ By: ________________________________________
Authorized Representative Authorized Representative
"""
    
    ner_model = NERModel()
    entities = ner_model.extract_entities(text)
    
    print("Extracted Entities:")
    for entity, label in entities:
        print(f"{entity} -> {label}")

if __name__ == "__main__":
    main()
    

    
