# agents/compliance_checker.py
from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime

from context_bank import ContextBank
from utils.ollama_client import OllamaClient
from utils.api_client import APIClient

from agents.tools.statutory_validator import validate_statutory_compliance, SeverityLevel
from agents.tools.precedent_analyzer import analyze_precedents_for_compliance
from agents.tools.consistency_engine import check_contractual_consistency
from agents.tools.hypergraph_analyzer import analyze_hypergraph_structure


def check_legal_compliance(
    clauses: List[Dict[str, Any]],
    document_metadata: Dict[str, Any],
    context_bank: ContextBank,
    knowledge_agent = None,
    use_ollama: bool = True,
    model_name: str = "llama3.1:latest",
    min_confidence: float = 0.75
) -> List[Dict[str, Any]]:
    """
    Checks legal document clauses for compliance issues and returns a structured list of non-compliant clauses.
    
    This function is designed to be called by the Compliance Checker agent in process.ipynb.
    It checks each clause against statutory regulations, legal precedents, and for internal consistency.
    
    Args:
        clauses: List of clause dictionaries, each containing at least 'id' and 'text' keys
        document_metadata: Metadata about the document (jurisdiction, document_type, etc.)
        context_bank: Shared context bank for storing and retrieving information
        knowledge_agent: Optional knowledge agent for retrieving legal information
        use_ollama: Whether to use Ollama for local model inference (default: True)
        model_name: Name of the model to use (default: "llama3.1:latest")
        min_confidence: Minimum confidence threshold for valid issues (default: 0.75)
        
    Returns:
        List[Dict[str, Any]]: List of non-compliant clauses with detailed analysis
    """
    # Initialize the appropriate LLM client
    if use_ollama:
        llm_client = OllamaClient(model_name)
    else:
        llm_client = APIClient(model_name)
        
    # Extract document information
    document_id = document_metadata.get("document_id", str(uuid.uuid4()))
    jurisdiction = document_metadata.get("jurisdiction", "US")
    document_type = document_metadata.get("document_type", "contract")
    
    # Prepare results container
    all_issues = []
    non_compliant_clauses = []
    
    # Process each clause for compliance issues
    for clause in clauses:
        clause_id = clause.get("id", str(uuid.uuid4()))
        clause_text = clause.get("text", "")
        
        if not clause_text:
            continue
            
        # Prepare document context for this clause
        document_context = {
            "document_id": document_id,
            "jurisdiction": jurisdiction,
            "document_type": document_type,
            "clauses": [c for c in clauses if c.get("id") != clause_id]
        }
        
        # Get relevant knowledge from knowledge agent if available
        knowledge_context = {}
        if knowledge_agent:
            try:
                # Retrieve relevant statutes
                statutes = knowledge_agent.find_relevant_statutes(
                    query=clause_text,
                    jurisdiction=jurisdiction
                )
                
                # Retrieve relevant precedents
                precedents = knowledge_agent.find_relevant_precedents(
                    query=clause_text,
                    jurisdiction=jurisdiction
                )
                
                knowledge_context = {
                    "statutes": statutes,
                    "precedents": precedents
                }
                
                # Store in context bank for future use
                context_bank.store(
                    key=f"knowledge:{clause_id}",
                    value=knowledge_context
                )
                
            except Exception as e:
                print(f"Error retrieving knowledge: {e}")
        
        # Add knowledge context to document context
        if knowledge_context:
            document_context["knowledge_context"] = knowledge_context
        
        # 1. Check statutory compliance
        statutory_violations = validate_statutory_compliance(
            clause_text=clause_text,
            llm_client=llm_client,
            clause_id=clause_id,
            jurisdiction=jurisdiction,
            knowledge_agent=knowledge_agent,
            min_confidence=min_confidence
        )
        
        # 2. Check precedent compliance
        precedent_context = knowledge_context.get("precedents", [])
        precedent_analysis = analyze_precedents_for_compliance(
            clause_text=clause_text,
            llm_client=llm_client,
            jurisdiction=jurisdiction,
            context={
                "precedents": precedent_context,
                **document_context
            } if precedent_context else document_context,
            min_confidence=min_confidence,
            use_web_search=False
        )
        precedent_issues = precedent_analysis.get("issues", [])
        
        # 3. Check contractual consistency
        # First, prepare the clauses list for consistency check
        current_clause = {"id": clause_id, "text": clause_text}
        all_clauses_for_consistency = [current_clause]
        
        if document_context.get("clauses"):
            for other_clause in document_context["clauses"]:
                if isinstance(other_clause, dict) and "text" in other_clause:
                    # Make sure each clause has an ID
                    if "id" not in other_clause:
                        other_clause["id"] = str(uuid.uuid4())
                    all_clauses_for_consistency.append(other_clause)
        
        # Perform consistency check
        consistency_analysis = check_contractual_consistency(
            clauses=all_clauses_for_consistency,
            llm_client=llm_client,
            document_context=document_context,
            min_confidence=min_confidence,
            use_hypergraph=True
        )
        
        # 4. Optional: Perform hypergraph analysis if there are enough clauses
        hypergraph_analysis = None
        if len(all_clauses_for_consistency) >= 3:
            hypergraph_analysis = analyze_hypergraph_structure(
                clauses=all_clauses_for_consistency,
                llm_client=llm_client,
                analyze_cycles=True,
                analyze_critical_nodes=True,
                analyze_clusters=True,
                node_to_analyze=clause_id
            )
        
        # Convert consistency issues to a standard format
        consistency_issues = []
        for inconsistency in consistency_analysis.get("inconsistencies", []):
            # Only include issues that involve the current clause
            if inconsistency.source_clause_id == clause_id or inconsistency.target_clause_id == clause_id:
                consistency_issues.append({
                    "description": inconsistency.description,
                    "severity": inconsistency.severity,
                    "reasoning": inconsistency.reasoning,
                    "references": [
                        f"Clause {inconsistency.source_clause_id}",
                        f"Clause {inconsistency.target_clause_id}"
                    ],
                    "implications": inconsistency.implications,
                    "confidence": inconsistency.confidence
                })
        
        # Combine all issues for this clause
        clause_issues = []
        
        # Add statutory violations
        for violation in statutory_violations:
            clause_issues.append({
                "type": "statutory",
                "description": violation.description,
                "severity": violation.severity.value,
                "references": [violation.statute_reference],
                "reasoning": violation.reasoning,
                "implications": violation.implications,
                "confidence": violation.confidence
            })
        
        # Add precedent issues
        for issue in precedent_issues:
            issue["type"] = "precedent"
            clause_issues.append(issue)
            
        # Add consistency issues
        for issue in consistency_issues:
            issue["type"] = "consistency"
            clause_issues.append(issue)
        
        # If there are any issues, add this clause to the non-compliant list
        if clause_issues:
            non_compliant_clause = {
                "clause_id": clause_id,
                "clause_text": clause_text,
                "issues": clause_issues,
                "issue_count": len(clause_issues),
                "statutory_violations": len(statutory_violations),
                "precedent_issues": len(precedent_issues),
                "consistency_issues": len(consistency_issues)
            }
            
            # Add hypergraph analysis if available
            if hypergraph_analysis:
                non_compliant_clause["hypergraph_analysis"] = {
                    "cycles": len(hypergraph_analysis["cycles"]),
                    "critical_nodes": len(hypergraph_analysis["critical_nodes"]),
                    "has_impact_analysis": hypergraph_analysis["impact_analysis"] is not None,
                    "relationship_clusters": len(hypergraph_analysis["relationship_clusters"])
                }
            
            # Add to our results list
            non_compliant_clauses.append(non_compliant_clause)
            
            # Store in context bank for use by other agents
            context_bank.store(
                key=f"compliance_analysis:{clause_id}",
                value={
                    "clause_id": clause_id,
                    "clause_text": clause_text,
                    "statutory_violations": [v.__dict__ for v in statutory_violations],
                    "precedent_issues": precedent_issues,
                    "consistency_issues": consistency_issues,
                    "hypergraph_analysis": hypergraph_analysis,
                    "all_issues": clause_issues,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    # Store document-level results in context bank
    document_analysis = {
        "document_id": document_id,
        "analysis_id": str(uuid.uuid4()),
        "clauses_analyzed": len(clauses),
        "non_compliant_clauses": len(non_compliant_clauses),
        "total_issues": sum(clause["issue_count"] for clause in non_compliant_clauses),
        "has_issues": len(non_compliant_clauses) > 0,
        "timestamp": datetime.now().isoformat()
    }
    
    context_bank.store(
        key=f"document_compliance_analysis:{document_id}",
        value=document_analysis
    )
    
    # Return the list of non-compliant clauses
    return non_compliant_clauses


# Legacy class maintained for backward compatibility
class ComplianceCheckerAgent:
    """
    Legacy Compliance Checker Agent class maintained for backward compatibility.
    New code should use the check_legal_compliance function instead.
    """
    
    def __init__(self, use_ollama: bool, model_name: str, context_bank: ContextBank, 
                 min_confidence: float = 0.75, knowledge_agent=None):
        self.use_ollama = use_ollama
        self.model_name = model_name
        self.context_bank = context_bank
        self.min_confidence = min_confidence
        self.knowledge_agent = knowledge_agent
        
        # Initialize client
        if use_ollama:
            self.llm_client = OllamaClient(model_name)
        else:
            self.llm_client = APIClient(model_name)
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document state to check for compliance issues."""
        # Extract document info from state
        document_id = state.get("document_id")
        if not document_id:
            state["error"] = "Missing document_id in state"
            state["next_step"] = "orchestrator"
            return state
            
        # Retrieve document and clauses from context bank
        document = self.context_bank.get_document(document_id)
        clauses = self.context_bank.clauses.get(document_id, [])
        
        if not document or not clauses:
            state["error"] = "Document or clauses not found in context bank"
            state["next_step"] = "orchestrator"
            return state
        
        # Create document metadata dict
        document_metadata = {
            "document_id": document_id,
            "jurisdiction": document.get("metadata", {}).get("jurisdiction", "US"),
            "document_type": document.get("metadata", {}).get("document_type", "contract")
        }
        
        # Call the new function to perform compliance checking
        non_compliant_clauses = check_legal_compliance(
            clauses=clauses,
            document_metadata=document_metadata,
            context_bank=self.context_bank,
            knowledge_agent=self.knowledge_agent,
            use_ollama=self.use_ollama,
            model_name=self.model_name,
            min_confidence=self.min_confidence
        )
        
        # Update state for next agent
        state["compliance_analyzed"] = True
        state["has_issues"] = len(non_compliant_clauses) > 0
        
        # Determine next step based on whether issues were found
        if state["has_issues"]:
            state["next_step"] = "clause_rewriter"  # Send to rewriter if there are issues
        else:
            state["next_step"] = "post_processor"   # Send to post-processor if no issues
            
        return state
