import re

class ClauseExtractor:
    def __init__(self):
        self.clause_patterns = [
            r"(?<=\n)\d+(\.\d+)*\s*-?\s*",  # Numbered clauses (e.g., "1.", "2.1", "3.2.1 -")
            r"(?i)\b(Article|Section|Clause)\s+\d+(\.\d+)*[:\.\s]",  # Legal terms like "Clause 5.1:"
            r"\([a-zA-Z]\)\s*",  # Bullet points like "(a)", "(i)", etc.
        ]

    def extract_clauses(self, text: str):
        clauses = []
        lines = text.split("\n")

        current_clause = ""
        for line in lines:
            if any(re.match(pattern, line) for pattern in self.clause_patterns):
                if current_clause:
                    clauses.append(current_clause.strip())  
                current_clause = line.strip()  
            else:
                if current_clause: 
                    current_clause += " " + line.strip()

        if current_clause:  
            clauses.append(current_clause.strip())

        return clauses if clauses else ["No clauses detected"]