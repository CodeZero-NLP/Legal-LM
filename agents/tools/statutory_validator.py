import time
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import uuid
from dataclasses import dataclass
from agents.utils.response_parser import ResponseParser
from agents.utils.confidence_scorer import ConfidenceScorer

STATUTORY_ANALYSIS_TEMPLATE = """
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

def format_statutory_prompt(clause_text: str, jurisdiction: str) -> str:
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

@dataclass
class Statute:
    id: str
    name: str
    section: str
    jurisdiction: str
    text: str
    url: Optional[str] = None

class SeverityLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class Violation:
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
    knowledge_context = None,  # if exists, assume it's a list of statutes
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
    
    print(f"Validating statutory compliance for clause {clause_id} in {jurisdiction} jurisdiction")
    
    response_parser = ResponseParser()
    confidence_scorer = ConfidenceScorer(min_confidence_threshold=min_confidence)
    # Assume knowledge_context is a list of statutes if provided
    relevant_statutes = knowledge_context if knowledge_context else []
    
    if relevant_statutes:
        print(f"Analyzing against {len(relevant_statutes)} relevant statutes")
    else:
        print("No specific statutes provided for analysis")
    
    # Step 2: Format statutory analysis prompt
    user_prompt = format_statutory_prompt(clause_text, jurisdiction)
    
    # Step 3: Get LLM analysis
    try:
        # Adding a delay here to avoid hitting API rate limits
        time.sleep(5)

        response = llm_client.query(
            system_prompt=STATUTORY_ANALYSIS_TEMPLATE,
            prompt=user_prompt
        )
    except Exception as e:
        print(f"ERROR: Failed to generate statutory analysis: {e}")
        return []
    
    # Step 4: Parse and score the response
    analysis_result = response_parser.parse_statutory_analysis(response)
    print(f"Identified {len(analysis_result)} potential statutory issues")
    
    violations = []
    for issue in analysis_result:
        confidence = confidence_scorer.score_issue(issue)  # get confidence using score_issue
        if confidence >= min_confidence:
            try:
                severity = SeverityLevel(issue.get("severity", "MEDIUM").upper())
            except ValueError:
                severity = SeverityLevel.MEDIUM
            
            violation = Violation(
                id=str(uuid.uuid4()),
                clause_id=clause_id,
                clause_text=clause_text,
                statute_reference=issue.get("statute_reference", "No specific statute referenced"),
                severity=severity,
                description=issue.get("description", ""),
                implications=issue.get("implications", []),
                reasoning=issue.get("reasoning", ""),
                confidence=confidence  # use computed confidence
            )
            violations.append(violation)
    
    print(f"Found {len(violations)} valid statutory violations with confidence >= {min_confidence}")
    return violations
