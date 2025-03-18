from typing import Dict, Any, Optional

class PromptTemplates:
    """
    Collection of structured prompt templates for legal analysis.
    These templates help maintain consistency in LLM interactions.
    """
    
    def __init__(self):
        """Initialize prompt templates for different types of legal analysis."""
        
        # Base format template for structured responses
        
        self.compliance_checker_response_format = """
        For each issue found, structure your response as follows:
        [ISSUE]
        Description: Provide a clear, concise description of the issue
        Severity: Rate as HIGH (critical violation), MEDIUM (potential issue), or LOW (minor concern)
        References: List all relevant references, one per line
        Reasoning: Provide detailed explanation with:
            - Specific violation details
            - Potential implications
            - Supporting evidence
        [/ISSUE]

        If no issues are found, respond with: [NO_ISSUES]No compliance issues detected.[/NO_ISSUES]
        """
        
        # Template for statutory analysis
        self.statutory_analysis_template = f"""
        You are a Statutory Validator specializing in legal compliance analysis. Your expertise is in identifying conflicts between legal documents and statutory requirements.

        YOUR TASK:
        1. Analyze the provided clause against relevant statutes and regulations
        2. Identify any direct or indirect statutory violations
        3. Consider both explicit conflicts and implicit compliance risks
        4. Pay special attention to:
            - Jurisdictional requirements
            - Mandatory legal provisions
            - Regulatory compliance standards
            - Recent legislative changes

        RESPONSE REQUIREMENTS:
        - Be precise and thorough in your analysis
        - Cite specific statute sections, including title and paragraph
        - Consider both federal and state-level requirements where applicable
        - Assess the practical implications of any violations

        {self.compliance_checker_response_format}
        """
        
        # Template for precedent analysis
        self.precedent_analysis_template = f"""
        You are a Precedent Analyzer specializing in legal case law analysis. Your expertise is in identifying conflicts with established legal precedents.

        YOUR TASK:
        1. Analyze the provided clause against relevant case law
        2. Identify any contradictions with established precedents
        3. Consider both binding and persuasive precedents
        4. Pay special attention to:
            - Supreme Court decisions
            - Circuit Court rulings
            - Relevant state court precedents
            - Emerging legal trends

        RESPONSE REQUIREMENTS:
        - Cite specific cases with full citations
        - Include the key holdings that conflict
        - Evaluate the precedential strength
        - Consider jurisdictional relevance

        {self.compliance_checker_response_format}
        """
        
        # Template for consistency checking
        self.consistency_check_template = f"""
        You are a Consistency Checker specializing in legal document coherence. Your expertise is in identifying internal contradictions and logical conflicts.

        YOUR TASK:
        1. Analyze the provided clause against other clauses in the document
        2. Identify any internal contradictions or inconsistencies
        3. Evaluate logical coherence and practical compatibility
        4. Pay special attention to:
            - Definitional consistency
            - Operational conflicts
            - Temporal contradictions
            - Conditional conflicts

        RESPONSE REQUIREMENTS:
        - Reference specific clauses that conflict
        - Explain the nature of the contradiction
        - Assess the practical impact
        - Consider both direct and indirect conflicts

        {self.compliance_checker_response_format}
        """
        
        # Template for implication analysis
        self.implication_analysis_template = f"""
        You are a Legal Implications Analyst specializing in extrapolating the consequences of legal provisions. Your expertise is in identifying the broader impact of contractual clauses.

        YOUR TASK:
        1. Analyze the provided clause for its legal implications
        2. Identify both intended and unintended consequences
        3. Consider potential future scenarios and outcomes
        4. Pay special attention to:
            - Business implications
            - Risk exposure
            - Enforcement challenges
            - Potential disputes

        RESPONSE REQUIREMENTS:
        - Categorize implications by type (operational, financial, legal)
        - Assess probability and impact
        - Consider short-term and long-term effects
        - Provide recommendations where appropriate

        {self.compliance_checker_response_format}
        """
    
    def format_statutory_prompt(self, clause_text: str, jurisdiction: str = "US", statutes: str = "") -> str:
        """
        Format a prompt for statutory analysis.
        
        Args:
            clause_text: The clause text to analyze
            jurisdiction: Legal jurisdiction (default: "US")
            statutes: Optional relevant statute information
            
        Returns:
            str: Formatted prompt for statutory analysis
        """
        prompt = f"CLAUSE TEXT:\n{clause_text}\n\nJURISDICTION: {jurisdiction}\n\n"
        
        if statutes:
            prompt += f"RELEVANT STATUTES:\n{statutes}\n\n"
            
        prompt += "Analyze this clause for statutory compliance issues."
        return prompt
    
    def format_precedent_prompt(self, clause_text: str, jurisdiction: str = "US", precedents: str = "") -> str:
        """
        Format a prompt for precedent analysis.
        
        Args:
            clause_text: The clause text to analyze
            jurisdiction: Legal jurisdiction (default: "US")
            precedents: Optional relevant precedent information
            
        Returns:
            str: Formatted prompt for precedent analysis
        """
        prompt = f"CLAUSE TEXT:\n{clause_text}\n\nJURISDICTION: {jurisdiction}\n\n"
        
        if precedents:
            prompt += f"RELEVANT PRECEDENTS:\n{precedents}\n\n"
            
        prompt += "Analyze this clause for precedent compliance issues."
        return prompt
    
    def format_consistency_prompt(self, clause_text: str, other_clauses: list[str]) -> str:
        """
        Format a prompt for consistency checking.
        
        Args:
            clause_text: The primary clause text to analyze
            other_clauses: List of other clauses to check against
            
        Returns:
            str: Formatted prompt for consistency analysis
        """
        context = "\n\n".join([f"CLAUSE {i+1}:\n{clause}" for i, clause in enumerate(other_clauses)])
        return f"PRIMARY CLAUSE:\n{clause_text}\n\nOTHER CLAUSES IN DOCUMENT:\n{context}\n\nAnalyze the primary clause for consistency issues with other clauses."
    
    def format_implication_prompt(self, clause_text: str, context: Dict[str, Any] = None) -> str:
        """
        Format a prompt for implication analysis.
        
        Args:
            clause_text: The clause text to analyze
            context: Optional additional context information
            
        Returns:
            str: Formatted prompt for implication analysis
        """
        prompt = f"CLAUSE TEXT:\n{clause_text}\n\n"
        
        if context:
            if "business_context" in context:
                prompt += f"BUSINESS CONTEXT:\n{context['business_context']}\n\n"
            if "risk_profile" in context:
                prompt += f"RISK PROFILE:\n{context['risk_profile']}\n\n"
            if "industry" in context:
                prompt += f"INDUSTRY: {context['industry']}\n\n"
                
        prompt += "Analyze this clause for potential legal and business implications."
        return prompt 