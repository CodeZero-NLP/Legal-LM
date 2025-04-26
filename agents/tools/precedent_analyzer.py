from typing import List, Dict, Any, Optional
from enum import Enum
import uuid
from dataclasses import dataclass, field

from agents.utils.response_parser import ResponseParser
from agents.utils.prompt_templates import PromptTemplates
from agents.utils.confidence_scorer import ConfidenceScorer

PRECEDENT_ANALYSIS_TEMPLATE = """
You are a legal precedent analyst specializing in case law implications.
Your task is to analyze a legal clause and determine if it conflicts with established legal precedents.

A precedent conflict exists when:
1. The clause contradicts rulings from relevant past cases
2. The clause attempts to enforce provisions that courts have ruled unenforceable
3. The clause uses approaches that courts have rejected
4. The clause would likely be invalidated based on existing precedents

Provide a clear analysis explaining:
- Whether a conflict with precedent exists
- The specific precedents involved (references)
- The severity of the conflict (HIGH, MEDIUM, LOW)
- The legal implications of the conflict
- Your reasoning process

Structure your response using the following format for each identified conflict:
[ISSUE]
Description: <description of the conflict>
Severity: <HIGH|MEDIUM|LOW>
References: <list of specific precedents involved>
Reasoning: <explanation of why it conflicts>
Implications: <potential legal consequences>
[/ISSUE]

If no conflicts are found, respond with:
[NO_ISSUES]
"""

# Reusing SeverityLevel definition, consider moving to a shared utils file later
class SeverityLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class PrecedentIssue:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    clause_id: Optional[str] = None
    clause_text: Optional[str] = None # Added for context
    description: str = ""
    severity: SeverityLevel = SeverityLevel.MEDIUM
    references: List[str] = field(default_factory=list)
    reasoning: str = ""
    implications: List[str] = field(default_factory=list) # Added field
    confidence: float = 0.0
    type: str = "precedent" # Added type field

def analyze_precedents_for_compliance(
    clause_text: str,
    llm_client: Any,
    jurisdiction: str,
    document_type: str, # Added document_type as per compliance_checker call
    knowledge_context: List[Dict[str, Any]], # Expecting list of precedent dicts
    min_confidence: float = 0.75,
    clause_id: Optional[str] = None # Added optional clause_id
) -> List[Dict[str, Any]]: # Return list of dicts as per parser output structure
    """
    Analyzes a clause for conflicts with legal precedents using an LLM.

    Args:
        clause_text: The text of the clause to analyze.
        llm_client: The language model client (Ollama or API).
        jurisdiction: The legal jurisdiction.
        document_type: The type of the document (e.g., contract, policy).
        knowledge_context: A list of dictionaries, each representing a relevant precedent.
                           Expected keys: 'title', 'content'.
        min_confidence: The minimum confidence score required to include an issue.
        clause_id: Optional identifier for the clause.

    Returns:
        A list of dictionaries, where each dictionary represents a potential precedent conflict issue.
    """
    if clause_id is None:
        clause_id = str(uuid.uuid4())

    response_parser = ResponseParser()
    confidence_scorer = ConfidenceScorer(min_confidence_threshold=min_confidence)
    prompt_templates = PromptTemplates()

    # Format precedent information for the prompt
    precedent_texts = []
    for precedent in knowledge_context:
        title = precedent.get('title', 'Unknown Precedent')
        content_excerpt = precedent.get('content', '')[:500] # Limit excerpt length
        precedent_texts.append(f"Title: {title}\nExcerpt: {content_excerpt}\n---")
    precedents_str = "\n".join(precedent_texts) if precedent_texts else "No specific precedents provided."

    # Format the user prompt
    user_prompt = prompt_templates.format_precedent_prompt(
        clause_text=clause_text,
        jurisdiction=jurisdiction,
        precedents=precedents_str
    )

    # Generate analysis using LLM
    try:
        response = llm_client.generate(
            system_prompt=PRECEDENT_ANALYSIS_TEMPLATE,
            user_prompt=user_prompt
        )
    except Exception as e:
        # Handle LLM client errors appropriately
        print(f"Error generating precedent analysis: {e}") # Replace with proper logging
        return []

    # Parse the LLM response
    analysis_result = response_parser.parse_precedent_analysis(response) # This returns a dict with 'issues' key
    parsed_issues = analysis_result.get("issues", [])

    # Score and filter issues
    valid_issues = []
    for issue_dict in parsed_issues:
        confidence = confidence_scorer.score_issue(issue_dict)
        if confidence >= min_confidence:
            try:
                severity = SeverityLevel(issue_dict.get("severity", "MEDIUM").upper())
            except ValueError:
                severity = SeverityLevel.MEDIUM

            # Add necessary fields and ensure structure matches expectations
            formatted_issue = {
                "id": str(uuid.uuid4()),
                "clause_id": clause_id,
                "clause_text": clause_text, # Add clause text for context
                "description": issue_dict.get("description", ""),
                "severity": severity.value,
                "references": issue_dict.get("references", []),
                "reasoning": issue_dict.get("reasoning", ""),
                "implications": issue_dict.get("implications", []), # Ensure implications are captured if provided by LLM
                "confidence": confidence,
                "type": "precedent"
            }
            valid_issues.append(formatted_issue)

    return valid_issues

