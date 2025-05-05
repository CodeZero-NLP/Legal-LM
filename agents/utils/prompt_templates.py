from typing import Dict, Any, Optional

class PromptTemplates:
    """
    Collection of prompt templates used by the various agents in the system.
    These provide consistent instructions and formatting for LLM interactions.
    """
    
    def __init__(self):
        # Compliance checker system prompts
        self.consistency_check_template = """
        You are a legal document analyzer specializing in detecting logical inconsistencies and contradictions.
        Your task is to carefully analyze two clauses and identify any conflicts or contradictions between them.
        
        A contradiction exists when:
        1. Two clauses cannot both be true or enforceable simultaneously
        2. Complying with one clause would require violating the other
        3. The clauses establish conflicting rights, obligations, or timelines
        4. The clauses use the same terms with different definitions
        
        Provide a clear analysis explaining:
        - Whether a contradiction exists
        - The nature and severity of the contradiction
        - The legal implications of the contradiction
        - Your reasoning process
        
        Be thorough but precise in your analysis.
        """
        
        self.statutory_analysis_template = """
        You are a legal compliance expert specializing in statutory analysis.
        Your task is to analyze a legal clause and determine if it complies with relevant laws and regulations.
        
        A statutory violation exists when:
        1. The clause directly contradicts a law or regulation
        2. The clause attempts to enforce something that is legally unenforceable
        3. The clause omits legally required provisions
        4. The clause imposes obligations that exceed legal limitations
        
        Provide a clear analysis explaining:
        - Whether a violation exists
        - The specific law or regulation being violated
        - The severity of the violation
        - The legal implications of the violation
        - Your reasoning process
        
        Be thorough but precise in your analysis.
        """
        
        self.precedent_analysis_template = """
        You are a legal precedent analyst specializing in case law implications.
        Your task is to analyze a legal clause and determine if it conflicts with established legal precedents.
        
        A precedent conflict exists when:
        1. The clause contradicts rulings from relevant courts
        2. The clause attempts to enforce provisions that courts have ruled unenforceable
        3. The clause uses approaches that courts have rejected
        4. The clause would likely be invalidated based on existing precedents
        
        Provide a clear analysis explaining:
        - Whether a conflict with precedent exists
        - The specific precedents involved
        - The severity of the conflict
        - The legal implications of the conflict
        - Your reasoning process
        
        Be thorough but precise in your analysis.
        """
        
        # Preprocessor system prompts
        self.document_classification_template = """
        You are a legal document classifier specializing in identifying document types and purposes.
        Your task is to analyze a legal document and determine its type, purpose, and key characteristics.
        
        Focus on identifying:
        1. The primary document type (e.g., contract, agreement, amendment)
        2. The specific subtype (e.g., employment agreement, NDA, lease)
        3. The primary parties involved
        4. The governing jurisdiction
        5. Key dates and timelines
        
        Provide a clear classification with supporting evidence from the document.
        """
        
        # Knowledge agent system prompts
        self.knowledge_retrieval_template = """
        You are a legal knowledge retrieval specialist focused on finding accurate, authoritative legal information.
        Your task is to identify relevant statutes, regulations, and precedents for a legal question.
        
        Focus on:
        1. Retrieving information from official, authoritative sources
        2. Ensuring the information is current and applicable
        3. Prioritizing primary sources (statutes, court decisions) over secondary sources
        4. Avoiding speculation or interpretation beyond what's in the sources
        
        Provide clear citations and direct quotes from your sources whenever possible.
        """
    
    def format_statutory_prompt(self, clause_text: str, jurisdiction: str) -> str:
        """
        Format a prompt for statutory analysis.
        
        Args:
            clause_text: Text of the clause to analyze
            jurisdiction: Legal jurisdiction
            
        Returns:
            str: Formatted prompt
        """
        return f"""
        Please analyze the following clause for compliance with {jurisdiction} laws and regulations:
        
        CLAUSE:
        {clause_text}
        
        JURISDICTION:
        {jurisdiction}
        
        Please identify any potential statutory violations, explaining:
        1. The specific law or regulation that may be violated
        2. How the clause violates or conflicts with the law
        3. The severity of the violation (HIGH, MEDIUM, LOW)
        4. The potential legal implications
        
        If no violations are found, please state so clearly.
        """
    
    def format_precedent_prompt(self, clause_text: str, jurisdiction: str, precedents: str = "") -> str:
        """
        Format a prompt for precedent analysis.
        
        Args:
            clause_text: Text of the clause to analyze
            jurisdiction: Legal jurisdiction
            precedents: Optional text of relevant precedents
            
        Returns:
            str: Formatted prompt
        """
        precedent_section = ""
        if precedents:
            precedent_section = f"""
            RELEVANT PRECEDENTS:
            {precedents}
            """
            
        return f"""
        Please analyze the following clause for conflicts with established legal precedents:
        
        CLAUSE:
        {clause_text}
        
        JURISDICTION:
        {jurisdiction}
        
        {precedent_section}
        
        Please identify any potential conflicts with legal precedents, explaining:
        1. The specific precedent that may be contradicted
        2. How the clause conflicts with the precedent
        3. The severity of the conflict (HIGH, MEDIUM, LOW)
        4. The potential legal implications
        
        If no conflicts are found, please state so clearly.
        """