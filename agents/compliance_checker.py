# agents/compliance_checker.py
from typing import Dict, List, Any, Optional
import uuid
from context_bank import ContextBank
from utils.ollama_client import OllamaClient
from utils.api_client import APIClient

class ComplianceCheckerAgent:
    """
    Compliance Checker Agent that detects contradictions between clauses
    and with relevant laws, provides reasoning, and extrapolates legal implications.
    """
    
    def __init__(self, use_ollama: bool, model_name: str, context_bank: ContextBank):
        """
        Initialize the Compliance Checker Agent.
        
        Args:
            use_ollama: Whether to use Ollama for local model inference
            model_name: Name of the model to use
            context_bank: Shared context bank for all agents
        """
        self.context_bank = context_bank
        
        # Initialize the appropriate client based on configuration
        if use_ollama:
            self.llm_client = OllamaClient(model_name)
        else:
            self.llm_client = APIClient(model_name)
        
        # System prompts for different compliance checking tasks
        self.statutory_validator_prompt = """
        You are a Statutory Validator specializing in legal compliance. Your task is to:
        1. Analyze the given clause against relevant statutes
        2. Identify any contradictions or compliance issues
        3. Provide detailed reasoning for each identified issue
        4. Cite the specific statute sections that are violated
        
        Be precise and thorough in your analysis.
        """
        
        self.precedent_analyzer_prompt = """
        You are a Precedent Analyzer specializing in legal case law. Your task is to:
        1. Analyze the given clause against relevant legal precedents
        2. Identify any contradictions with established case law
        3. Provide detailed reasoning for each identified issue
        4. Cite the specific cases that establish contrary precedent
        
        Be precise and thorough in your analysis.
        """
        
        self.consistency_checker_prompt = """
        You are a Consistency Checker specializing in internal document consistency. Your task is to:
        1. Analyze the given clause against other clauses in the document
        2. Identify any internal contradictions or inconsistencies
        3. Provide detailed reasoning for each identified issue
        
        Be precise and thorough in your analysis.
        """
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and perform compliance checks.
        
        Args:
            state: Current state of the workflow
            
        Returns:
            Dict: Updated state with compliance check results
        """
        document_id = state.get("document_id")
        clauses = self.context_bank.clauses.get(document_id, [])
        
        # Initialize a list to store detected contradictions
        contradictions = []
        
        # Iterate over each clause to perform compliance checks
        for clause in clauses:
            clause_id = clause.get("id")
            clause_text = clause.get("text")
            
            # Perform statutory validation
            statutory_issues = self._validate_statutory_compliance(clause_text)
            if statutory_issues:
                contradictions.append({
                    "clause_id": clause_id,
                    "type": "statutory",
                    "issues": statutory_issues
                })
            
            # Perform precedent analysis
            precedent_issues = self._analyze_precedent_compliance(clause_text)
            if precedent_issues:
                contradictions.append({
                    "clause_id": clause_id,
                    "type": "precedent",
                    "issues": precedent_issues
                })
            
            # Perform internal consistency check
            consistency_issues = self._check_internal_consistency(clause_text, clauses)
            if consistency_issues:
                contradictions.append({
                    "clause_id": clause_id,
                    "type": "consistency",
                    "issues": consistency_issues
                })
        
        # Update the state with detected contradictions
        state["contradictions_found"] = bool(contradictions)
        state["contradictions"] = contradictions
        return state
    
    def _validate_statutory_compliance(self, clause_text: str) -> List[Dict[str, Any]]:
        """
        Validate the clause against relevant statutes.
        
        Args:
            clause_text: Text of the clause to validate
            
        Returns:
            List: Detected statutory issues
        """
        # Prepare the input for the LLM
        input_text = f"Clause: {clause_text}\n\nAnalyze for statutory compliance."
        
        # Get statutory validation from LLM
        response = self.llm_client.generate(
            system_prompt=self.statutory_validator_prompt,
            user_prompt=input_text
        )
        
        # Parse the response to extract issues
        issues = self._parse_issues(response)
        return issues
    
    def _analyze_precedent_compliance(self, clause_text: str) -> List[Dict[str, Any]]:
        """
        Analyze the clause against relevant legal precedents.
        
        Args:
            clause_text: Text of the clause to analyze
            
        Returns:
            List: Detected precedent issues
        """
        # Prepare the input for the LLM
        input_text = f"Clause: {clause_text}\n\nAnalyze for precedent compliance."
        
        # Get precedent analysis from LLM
        response = self.llm_client.generate(
            system_prompt=self.precedent_analyzer_prompt,
            user_prompt=input_text
        )
        
        # Parse the response to extract issues
        issues = self._parse_issues(response)
        return issues
    
    def _check_internal_consistency(self, clause_text: str, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check the clause for internal consistency within the document.
        
        Args:
            clause_text: Text of the clause to check
            clauses: List of all clauses in the document
            
        Returns:
            List: Detected consistency issues
        """
        # Prepare the input for the LLM
        input_text = f"Clause: {clause_text}\n\nCheck for internal consistency within the document."
        
        # Get consistency check from LLM
        response = self.llm_client.generate(
            system_prompt=self.consistency_checker_prompt,
            user_prompt=input_text
        )
        
        # Parse the response to extract issues
        issues = self._parse_issues(response)
        return issues
    
    def _parse_issues(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response to extract issues.
        
        Args:
            response: Response text from the LLM
            
        Returns:
            List: Parsed issues
        """
        # This is a simplified implementation - in a real system, you'd want more robust parsing
        issues = []
        for line in response.strip().split('\n'):
            if line.startswith('- '):
                issue_desc = line[2:].strip()
                issues.append({
                    "description": issue_desc
                })
        return issues
