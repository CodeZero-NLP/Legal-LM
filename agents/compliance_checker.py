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


# Rewrite the check_legal_compliance function.
# It should only take the following arguments:
# context bank, knowledge_from_vector_db, use_ollama as a bool, model_name, min_confidence as a float
# The document metadata as well as the complete document should be obtained 
# from the context bank along with the clauses, the entities and, the laws.
# The knowledge_from_vector_db will include all the information retrieved from the vector database


def check_legal_compliance(
    context_bank: ContextBank,
    knowledge_from_vector_db: List[Dict[str, Any]],
    use_ollama: bool = False,
    model_name: str = "gemini-2.0-flash",
    min_confidence: float = 0.75
) -> List[Dict[str, Any]]:
    """
    Checks legal document clauses for compliance issues and returns a structured list of non-compliant clauses.
    
    This function is designed to be called by the Compliance Checker agent in process.ipynb.
    It checks each clause against statutory regulations, legal precedents, and for internal consistency.
    
    Args:
        context_bank: Shared context bank containing document, clauses, entities, and laws
        knowledge_from_vector_db: Information retrieved from the vector database using search_in_qdrant
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
        llm_client = APIClient("gemini-2.0-flash")
        
    
    # Find the active document ID (assuming it's stored or retrievable)
    document_ids = list(context_bank.documents.keys())
    if not document_ids:
        return []  # No documents to analyze
    
    # For simplicity, use the first document if multiple exist
    # In a production environment, you might want to analyze all or use a specific one
    document_id = document_ids[0]
    
    # Extract document information from context bank
    document = context_bank.get_document(document_id)
    if not document:
        return []  # Document not found
    
    # Get document content
    document_content = document.get("content", "")
    if not document_content:
        return []  # Empty document
    
    # Use LLM to estimate jurisdiction and document type instead of relying on metadata
    doc_analysis_prompt = f"""
    Analyze the following legal document excerpt. Determine:
    1. The most likely jurisdiction (e.g., US, UK, EU, etc.)
    2. The type of legal document (e.g., contract, policy, regulation, etc.)
    
    Format your response as a JSON with keys 'jurisdiction' and 'document_type'.
    
    Document excerpt:
    {document_content[:2000]}  # Use first 2000 chars for analysis
    """
    
    doc_analysis_response = llm_client.query(doc_analysis_prompt)
    
    # Extract jurisdiction and document type from LLM response
    # Default to US jurisdiction and contract type if parsing fails
    try:
        import json
        doc_analysis = json.loads(doc_analysis_response)
        jurisdiction = doc_analysis.get("jurisdiction", "US")
        document_type = doc_analysis.get("document_type", "contract")
    except:
        # Default values if parsing fails
        jurisdiction = "US"
        document_type = "contract"
    
    # Get clauses from context bank
    clauses = context_bank.clauses.get(document_id, [])
    if not clauses:
        return []  # No clauses to analyze
    
    # Get entities from context bank
    entities = context_bank.entities.get(document_id, [])
    
    # Get laws from context bank
    laws = {}
    for law_id, law_data in context_bank.laws.items():
        if law_data.get("metadata", {}).get("document_id") == document_id:
            laws[law_id] = law_data
    
    # Prepare results container
    all_issues = []
    non_compliant_clauses = []
    
    # Prepare knowledge context from vector database
    # Initialize with empty lists to handle the case where vector DB has no statutes/precedents
    knowledge_context = {
        "vector_db_results": knowledge_from_vector_db,
        "statutes": [],
        "precedents": []
    }
    
    # Only attempt to extract statutes and precedents if knowledge_from_vector_db has content
    if knowledge_from_vector_db:
        # Use LLM to classify each knowledge item as statute or precedent
        for item in knowledge_from_vector_db:
            content = item.get("content", "")
            title = item.get("title", "")
            
            if not content and not title:
                continue
            
            # Use LLM to classify the knowledge item
            classification_prompt = f"""
            Determine if the following legal text is more likely to be a statute/regulation or a legal precedent/case law.
            
            Title: {title}
            Content excerpt: {content[:500]}
            
            Respond with only one word: either "statute" or "precedent".
            """
            
            classification = llm_client.query(classification_prompt).strip().lower()
            
            if "statute" in classification or "regulation" in classification or "code" in classification:
                knowledge_context["statutes"].append({
                    "title": title,
                    "content": content,
                    "url": item.get("url", ""),
                    "score": item.get("score", 0.0)
                })
            else:
                knowledge_context["precedents"].append({
                    "title": title,
                    "content": content,
                    "url": item.get("url", ""),
                    "score": item.get("score", 0.0)
                })
    
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
            "clauses": [c for c in clauses if c.get("id") != clause_id],
            "entities": entities,
            "knowledge_context": knowledge_context
        }
        
        # 1. Check statutory compliance - only if statutes are available
        statutory_violations = []
        if knowledge_context["statutes"]:
            statutory_violations = validate_statutory_compliance(
                clause_text=clause_text,
                llm_client=llm_client,
                clause_id=clause_id,
                jurisdiction=jurisdiction,
                knowledge_context=knowledge_context["statutes"],
                min_confidence=min_confidence
            )
        
        # 2. Check precedent compliance - only if precedents are available
        precedent_issues = []
        if knowledge_context["precedents"]:
            precedent_context = knowledge_context["precedents"]
            precedent_analysis = analyze_precedents_for_compliance(
                clause_text=clause_text,
                llm_client=llm_client,
                jurisdiction=jurisdiction,
                context={
                    "precedents": precedent_context,
                    **document_context
                },
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
        "timestamp": datetime.now().isoformat(),
        "estimated_jurisdiction": jurisdiction,
        "estimated_document_type": document_type
    }
    
    context_bank.store(
        key=f"document_compliance_analysis:{document_id}",
        value=document_analysis
    )
    
    # Return the list of non-compliant clauses
    return non_compliant_clauses
