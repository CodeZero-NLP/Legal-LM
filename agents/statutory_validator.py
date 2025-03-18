from typing import Dict, List, Any, Optional, Union
from enum import Enum
import uuid
from dataclasses import dataclass

from utils.prompt_templates import PromptTemplates
from utils.response_parser import ResponseParser
from utils.confidence_scorer import ConfidenceScorer


class SeverityLevel(str, Enum):
    """Enum for violation severity levels"""
    HIGH = "HIGH"       # Critical legal violation
    MEDIUM = "MEDIUM"   # Potential legal issue
    LOW = "LOW"         # Minor concern


@dataclass
class Statute:
    """Representation of a statutory reference"""
    id: str
    name: str
    section: str
    jurisdiction: str
    text: str
    url: Optional[str] = None


@dataclass
class Violation:
    """Representation of a statutory violation"""
    id: str
    clause_id: str
    statute_reference: str
    severity: SeverityLevel
    description: str
    implications: List[str]
    reasoning: str
    confidence: float


class StatutoryValidator:
    """
    Validates clauses against statutory laws by leveraging LLM-based analysis
    and structured knowledge of legal statutes.
    
    This component identifies potential violations of statutory requirements,
    assesses their severity, and provides detailed reasoning about the legal implications.
    """
    
    def __init__(self, llm_client, knowledge_agent=None, min_confidence: float = 0.75):
        """
        Initialize the statutory validator.
        
        Args:
            llm_client: Client for LLM interactions (Ollama or API)
            knowledge_agent: Optional knowledge agent for statute retrieval
            min_confidence: Minimum confidence threshold for valid issues
        """
        self.llm_client = llm_client
        self.knowledge_agent = knowledge_agent
        self.min_confidence = min_confidence
        
        # Initialize utility components
        self.prompt_templates = PromptTemplates()
        self.response_parser = ResponseParser()
        self.confidence_scorer = ConfidenceScorer(min_confidence_threshold=min_confidence)
    
    def validate_clause(self, clause_text: str, clause_id: str = None, jurisdiction: str = "US") -> List[Violation]:
        """
        Validate a clause against relevant statutory laws.
        
        This method:
        1. Retrieves relevant statutes if knowledge agent is available
        2. Formats a prompt for the LLM to analyze statutory compliance
        3. Processes the LLM response to extract structured violations
        4. Filters out low-confidence results
        5. Converts issues to formal Violation objects
        
        Args:
            clause_text: Text of the clause to validate
            clause_id: Optional identifier for the clause
            jurisdiction: Legal jurisdiction (default: "US")
            
        Returns:
            List[Violation]: List of detected statutory violations
        """
        if clause_id is None:
            clause_id = str(uuid.uuid4())
        
        # Step 1: Retrieve relevant statutes if knowledge agent is available
        relevant_statutes = []
        if self.knowledge_agent:
            clause_context = {"text": clause_text, "jurisdiction": jurisdiction}
            relevant_statutes = self.get_relevant_statutes(clause_context)
            
        # Step 2: Format statutory analysis prompt
        statute_context = self._format_statute_context(relevant_statutes)
        user_prompt = self.prompt_templates.format_statutory_prompt(clause_text, jurisdiction)
        
        # If we have relevant statutes, add them to the prompt
        if statute_context:
            user_prompt = f"{user_prompt}\n\nRELEVANT STATUTES:\n{statute_context}"
        
        # Step 3: Get LLM analysis
        response = self.llm_client.generate(
            system_prompt=self.prompt_templates.statutory_analysis_template,
            user_prompt=user_prompt
        )
        
        # Step 4: Parse and score the response
        analysis_result = self.response_parser.parse_statutory_analysis(response)
        scored_analysis = self.confidence_scorer.score_analysis(analysis_result)
        
        # Step 5: Convert high-confidence issues to Violation objects
        violations = []
        for issue in self.confidence_scorer.get_high_confidence_issues(scored_analysis):
            severity = self._normalize_severity(issue.get("severity", "MEDIUM"))
            
            # Extract implications from reasoning
            implications = self._extract_implications(issue.get("reasoning", ""))
            
            # Create violation object
            violation = Violation(
                id=str(uuid.uuid4()),
                clause_id=clause_id,
                statute_reference=self._extract_primary_reference(issue.get("references", [])),
                severity=severity,
                description=issue.get("description", ""),
                implications=implications,
                reasoning=issue.get("reasoning", ""),
                confidence=issue.get("confidence", 0.0)
            )
            
            violations.append(violation)
        
        return violations
    
    def assess_severity(self, violation: Union[Violation, Dict]) -> SeverityLevel:
        """
        Assess or reassess the severity of a violation.
        
        This method can be used to validate the LLM-assigned severity
        or to reassess it based on additional context.
        
        Args:
            violation: Violation object or issue dictionary
            
        Returns:
            SeverityLevel: Assessed severity level
        """
        # Extract severity string depending on input type
        if isinstance(violation, Violation):
            severity_str = violation.severity
            reasoning = violation.reasoning
        else:
            severity_str = violation.get("severity", "MEDIUM")
            reasoning = violation.get("reasoning", "")
        
        # Simple heuristics to adjust severity based on content
        # These could be expanded with more sophisticated logic
        if any(term in reasoning.lower() for term in ["illegal", "criminal", "liability", "void", "penalty"]):
            return SeverityLevel.HIGH
        elif any(term in reasoning.lower() for term in ["ambiguous", "unclear", "may not", "could be"]):
            return SeverityLevel.MEDIUM
        
        # Default to the provided severity or MEDIUM if invalid
        return self._normalize_severity(severity_str)
    
    def get_relevant_statutes(self, clause_context: Dict) -> List[Statute]:
        """
        Retrieve statutes relevant to the clause context.
        
        Args:
            clause_context: Dictionary with clause information
            
        Returns:
            List[Statute]: List of relevant statutes
        """
        if not self.knowledge_agent:
            return []
        
        # Use the knowledge agent to retrieve relevant statutes
        # This is a simplified implementation - in a real system,
        # there would be more sophisticated integration with the knowledge agent
        try:
            raw_statutes = self.knowledge_agent.find_relevant_statutes(
                query=clause_context.get("text", ""),
                jurisdiction=clause_context.get("jurisdiction", "US")
            )
            
            # Convert raw results to Statute objects
            statutes = []
            for raw in raw_statutes:
                statute = Statute(
                    id=raw.get("id", str(uuid.uuid4())),
                    name=raw.get("name", ""),
                    section=raw.get("section", ""),
                    jurisdiction=raw.get("jurisdiction", "US"),
                    text=raw.get("text", ""),
                    url=raw.get("url")
                )
                statutes.append(statute)
            
            return statutes
        except Exception as e:
            # Log the error and return empty list
            print(f"Error retrieving statutes: {e}")
            return []
    
    def _format_statute_context(self, statutes: List[Statute]) -> str:
        """
        Format a list of statutes into a string context for the LLM prompt.
        
        Args:
            statutes: List of Statute objects
            
        Returns:
            str: Formatted statute context
        """
        if not statutes:
            return ""
        
        context_parts = []
        for statute in statutes:
            context_parts.append(
                f"Statute: {statute.name}\n"
                f"Section: {statute.section}\n"
                f"Jurisdiction: {statute.jurisdiction}\n"
                f"Text: {statute.text}\n"
            )
        
        return "\n\n".join(context_parts)
    
    def _normalize_severity(self, severity_str: str) -> SeverityLevel:
        """
        Normalize a severity string to a valid SeverityLevel.
        
        Args:
            severity_str: String representation of severity
            
        Returns:
            SeverityLevel: Normalized severity level
        """
        try:
            return SeverityLevel(severity_str.upper())
        except ValueError:
            # Default to MEDIUM if the severity isn't recognized
            return SeverityLevel.MEDIUM
    
    def _extract_implications(self, reasoning: str) -> List[str]:
        """
        Extract implications from reasoning text.
        
        Args:
            reasoning: Reasoning text from the LLM
            
        Returns:
            List[str]: Extracted implications
        """
        implications = []
        
        # Look for implications in the reasoning text
        # This is a simple implementation that could be expanded
        if "implication" in reasoning.lower() or "consequence" in reasoning.lower():
            # Split into sentences and look for implication sentences
            sentences = reasoning.split(". ")
            for sentence in sentences:
                if any(term in sentence.lower() for term in ["implication", "consequence", "result", "lead to", "could cause"]):
                    implications.append(sentence.strip())
        
        # If no specific implications found, use a generic one
        if not implications and reasoning:
            implications.append("Potential compliance issue identified based on analysis.")
        
        return implications
    
    def _extract_primary_reference(self, references: List[str]) -> str:
        """
        Extract the primary statutory reference from a list of references.
        
        Args:
            references: List of reference strings
            
        Returns:
            str: Primary reference or default message
        """
        if not references:
            return "No specific statute referenced"
        
        # For now, simply return the first reference
        # In a more sophisticated implementation, this could identify
        # the most relevant or authoritative reference
        return references[0] 