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


def validate_statutory_compliance(
    clause_text: str, 
    llm_client, 
    clause_id: str = None, 
    jurisdiction: str = "US", 
    knowledge_agent = None, 
    min_confidence: float = 0.75
) -> List[Violation]:
    """
    Validate a clause against relevant statutory laws.
    
    This function:
    1. Retrieves relevant statutes if knowledge agent is available
    2. Formats a prompt for the LLM to analyze statutory compliance
    3. Processes the LLM response to extract structured violations
    4. Filters out low-confidence results
    5. Converts issues to formal Violation objects
    
    Args:
        clause_text: Text of the clause to validate
        llm_client: Client for LLM interactions (Ollama or API)
        clause_id: Optional identifier for the clause (default: generated UUID)
        jurisdiction: Legal jurisdiction (default: "US")
        knowledge_agent: Optional knowledge agent for statute retrieval
        min_confidence: Minimum confidence threshold for valid issues (default: 0.75)
        
    Returns:
        List[Violation]: List of detected statutory violations
    """
    if clause_id is None:
        clause_id = str(uuid.uuid4())
    
    # Initialize utility components
    prompt_templates = PromptTemplates()
    response_parser = ResponseParser()
    confidence_scorer = ConfidenceScorer(min_confidence_threshold=min_confidence)
    
    # Helper function: Format statute context
    def format_statute_context(statutes: List[Statute]) -> str:
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
    
    # Helper function: Normalize severity
    def normalize_severity(severity_str: str) -> SeverityLevel:
        try:
            return SeverityLevel(severity_str.upper())
        except ValueError:
            # Default to MEDIUM if the severity isn't recognized
            return SeverityLevel.MEDIUM
    
    # Helper function: Extract implications
    def extract_implications(reasoning: str) -> List[str]:
        implications = []
        
        # Look for implications in the reasoning text
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
    
    # Helper function: Extract primary reference
    def extract_primary_reference(references: List[str]) -> str:
        if not references:
            return "No specific statute referenced"
        
        # For now, simply return the first reference
        return references[0]
    
    # Helper function: Get relevant statutes
    def get_relevant_statutes(clause_context: Dict) -> List[Statute]:
        if not knowledge_agent:
            return []
        
        # Use the knowledge agent to retrieve relevant statutes
        try:
            raw_statutes = knowledge_agent.find_relevant_statutes(
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
    
    # Step 1: Retrieve relevant statutes if knowledge agent is available
    relevant_statutes = []
    if knowledge_agent:
        clause_context = {"text": clause_text, "jurisdiction": jurisdiction}
        relevant_statutes = get_relevant_statutes(clause_context)
    
    # Step 2: Format statutory analysis prompt
    statute_context = format_statute_context(relevant_statutes)
    user_prompt = prompt_templates.format_statutory_prompt(clause_text, jurisdiction)
    
    # If we have relevant statutes, add them to the prompt
    if statute_context:
        user_prompt = f"{user_prompt}\n\nRELEVANT STATUTES:\n{statute_context}"
    
    # Step 3: Get LLM analysis
    response = llm_client.generate(
        system_prompt=prompt_templates.statutory_analysis_template,
        user_prompt=user_prompt
    )
    
    # Step 4: Parse and score the response
    analysis_result = response_parser.parse_statutory_analysis(response)
    scored_analysis = confidence_scorer.score_analysis(analysis_result)
    
    # Step 5: Convert high-confidence issues to Violation objects
    violations = []
    for issue in confidence_scorer.get_high_confidence_issues(scored_analysis):
        severity = normalize_severity(issue.get("severity", "MEDIUM"))
        
        # Extract implications from reasoning
        implications = extract_implications(issue.get("reasoning", ""))
        
        # Create violation object
        violation = Violation(
            id=str(uuid.uuid4()),
            clause_id=clause_id,
            statute_reference=extract_primary_reference(issue.get("references", [])),
            severity=severity,
            description=issue.get("description", ""),
            implications=implications,
            reasoning=issue.get("reasoning", ""),
            confidence=issue.get("confidence", 0.0)
        )
        
        violations.append(violation)
    
    return violations


class StatutoryValidator:
    """
    Legacy class maintained for backward compatibility.
    Validates clauses against statutory laws using the validate_statutory_compliance function.
    """
    
    def __init__(self, llm_client, knowledge_agent=None, min_confidence: float = 0.75):
        self.llm_client = llm_client
        self.knowledge_agent = knowledge_agent
        self.min_confidence = min_confidence
        self.prompt_templates = PromptTemplates()
        self.response_parser = ResponseParser()
        self.confidence_scorer = ConfidenceScorer(min_confidence_threshold=min_confidence)
    
    def validate_clause(self, clause_text: str, clause_id: str = None, jurisdiction: str = "US") -> List[Violation]:
        """
        Validate a clause against relevant statutory laws.
        This method now simply calls the standalone validate_statutory_compliance function.
        
        Args:
            clause_text: Text of the clause to validate
            clause_id: Optional identifier for the clause
            jurisdiction: Legal jurisdiction (default: "US")
            
        Returns:
            List[Violation]: List of detected statutory violations
        """
        return validate_statutory_compliance(
            clause_text=clause_text,
            llm_client=self.llm_client,
            clause_id=clause_id,
            jurisdiction=jurisdiction,
            knowledge_agent=self.knowledge_agent,
            min_confidence=self.min_confidence
        )
    
    # Legacy methods maintained for backward compatibility
    def assess_severity(self, violation: Union[Violation, Dict]) -> SeverityLevel:
        """Assess or reassess the severity of a violation."""
        # Extract severity string depending on input type
        if isinstance(violation, Violation):
            severity_str = violation.severity
            reasoning = violation.reasoning
        else:
            severity_str = violation.get("severity", "MEDIUM")
            reasoning = violation.get("reasoning", "")
        
        # Simple heuristics to adjust severity based on content
        if any(term in reasoning.lower() for term in ["illegal", "criminal", "liability", "void", "penalty"]):
            return SeverityLevel.HIGH
        elif any(term in reasoning.lower() for term in ["ambiguous", "unclear", "may not", "could be"]):
            return SeverityLevel.MEDIUM
        
        # Default to the provided severity or MEDIUM if invalid
        try:
            return SeverityLevel(severity_str.upper())
        except ValueError:
            return SeverityLevel.MEDIUM
    
    def get_relevant_statutes(self, clause_context: Dict) -> List[Statute]:
        """Retrieve statutes relevant to the clause context."""
        if not self.knowledge_agent:
            return []
        
        try:
            raw_statutes = self.knowledge_agent.find_relevant_statutes(
                query=clause_context.get("text", ""),
                jurisdiction=clause_context.get("jurisdiction", "US")
            )
            
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
            print(f"Error retrieving statutes: {e}")
            return []