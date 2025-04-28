# agents/compliance_checker.py
import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from context_bank import ContextBank
from agents.utils.ollama_client import OllamaClient
from agents.utils.api_client import APIClient

from agents.tools.statutory_validator import validate_statutory_compliance, SeverityLevel
from agents.tools.precedent_analyzer import analyze_precedents_for_compliance
from agents.tools.consistency_engine import check_contractual_consistency
from agents.tools.hypergraph_analyzer import analyze_hypergraph_structure

def check_legal_compliance(
    context_bank: ContextBank,
    knowledge_from_vector_db: List[Dict[str, Any]],
    use_ollama: bool = False,
    model_name: str = "gemini-2.0-flash",
    min_confidence: float = 0.75
) -> List[Dict[str, Any]]:
    """
    Checks legal document clauses for compliance issues and returns a structured list of non-compliant clauses.
    
    This function analyzes each clause against statutory regulations, legal precedents, and for internal consistency,
    and returns a comprehensive compliance analysis.
    
    Args:
        context_bank: Shared context bank containing document, clauses, entities, and jurisdiction
        knowledge_from_vector_db: Information retrieved from the vector database containing relevant legal knowledge
        use_ollama: Whether to use Ollama for local model inference (default: False)
        model_name: Name of the model to use (default: "gemini-2.0-flash")
        min_confidence: Minimum confidence threshold for valid issues (default: 0.75)
        
    Returns:
        List[Dict[str, Any]]: List of non-compliant clauses with detailed analysis
    """


    # Start time tracking for performance analysis
    start_time = datetime.now()
    print(f"Starting legal compliance check with model '{model_name}' (use_ollama={use_ollama})")
    
    # Initialize the appropriate LLM client
    llm_client = _initialize_llm_client(use_ollama, model_name)
    if not llm_client:
        print("ERROR: Failed to initialize LLM client")
        return []
    
    # Get document from context bank
    document = context_bank.get_document()
    if not document:
        print("ERROR: No document found in context bank")
        return []
    
    document_content = document.get("content", "")
    if not document_content:
        print("ERROR: Document has no content")
        return []
    
    # Get jurisdiction from context bank or estimate using document content
    jurisdiction = context_bank.get_jurisdiction()

    # Adding a delay here to avoid hitting API rate limits
    time.sleep(5)

    if not jurisdiction:
        print("WARNING: No jurisdiction found in context bank, estimating from document content")
        jurisdiction = _estimate_jurisdiction(document_content, llm_client)
        print(f"Estimated jurisdiction: {jurisdiction}")
        # Store the estimated jurisdiction
        context_bank.add_jurisdiction(jurisdiction)
    
    # Get document type (estimate if necessary)
    document_type = document.get("metadata", {}).get("document_type")

    # Adding a delay here to avoid hitting API rate limits
    time.sleep(5)

    if not document_type:
        print("WARNING: No document type found in metadata, estimating from document content")
        document_type = _estimate_document_type(document_content, llm_client)
        print(f"Estimated document type: {document_type}")
    
    # Get clauses from context bank
    clauses = context_bank.get_clauses()
    if not clauses:
        print("ERROR: No clauses found in context bank")
        return []
    
    # Get entities from context bank
    entities = context_bank.get_entities()
    print(f"Found {len(clauses)} clauses and {len(entities)} entities")
    
    # Adding a delay here to avoid hitting API rate limits
    time.sleep(5)

    # Prepare knowledge context from vector database
    knowledge_context = _prepare_knowledge_context(knowledge_from_vector_db, llm_client)
    print(f"Prepared knowledge context with {len(knowledge_context['statutes'])} statutes and {len(knowledge_context['precedents'])} precedents")
    
    # Process each clause for compliance issues
    non_compliant_clauses = []
    for i, clause in enumerate(clauses):
        print(f"Analyzing clause {i+1}/{len(clauses)} (ID: {clause.get('id', 'unknown')})")
        
        # Updated to use 'Text' instead of 'text'
        print(f"Clause Text: {clause.get('Text', 'No text available')}")

        # Updated to use 'Category' instead of 'category'
        print(f"Clause Category: {clause.get('Category', 'No category available')}")


        # Adding a delay here to avoid hitting API rate limits
        time.sleep(5)

        clause_analysis = _analyze_clause_compliance(
            clause=clause,
            context_bank=context_bank,
            all_clauses=clauses,
            entities=entities,
            jurisdiction=jurisdiction,
            document_type=document_type,
            knowledge_context=knowledge_context,
            llm_client=llm_client,
            min_confidence=min_confidence
        )
        
        # If there are any issues, add this clause to the non-compliant list
        if clause_analysis["has_issues"]:
            non_compliant_clauses.append(clause_analysis["result"])
            print(f"Found {clause_analysis['issue_count']} compliance issues in clause")
    
    # Store document-level results in context bank
    _store_document_analysis(
        context_bank=context_bank,
        clauses_analyzed=len(clauses),
        non_compliant_clauses=non_compliant_clauses,
        jurisdiction=jurisdiction,
        document_type=document_type
    )
    
    # Log performance metrics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Completed compliance check in {duration:.2f} seconds. Found {len(non_compliant_clauses)} non-compliant clauses")
    
    return non_compliant_clauses


def _initialize_llm_client(use_ollama: bool, model_name: str) -> Any:
    """Initialize and return the appropriate LLM client based on settings."""
    try:
        if use_ollama:
            print(f"Initializing Ollama client with model {model_name}")
            return OllamaClient(model_name)
        else:
            print(f"Initializing API client with model {model_name}")
            return APIClient(model_name)
    except Exception as e:
        print(f"ERROR: Failed to initialize LLM client: {str(e)}")
        return None


def _estimate_jurisdiction(document_content: str, llm_client: Any) -> str:
    """Estimate the jurisdiction from document content using LLM."""
    # Take the first 2000 characters for analysis to avoid token limits
    content_sample = document_content[:2000]
    prompt = f"""
    Analyze the following legal document excerpt and determine the most likely jurisdiction.
    Provide only the jurisdiction name (e.g., "US", "California", "UK", "EU").
    
    Document excerpt:
    {content_sample}
    """
    
    try:
        response = llm_client.query(prompt)
        # Clean the response to get just the jurisdiction name
        jurisdiction = response.strip().split('\n')[0].strip()
        return jurisdiction or "US"  # Default to US if empty
    except Exception as e:
        print(f"ERROR: Error estimating jurisdiction: {str(e)}")
        return "US"  # Default to US on error


def _estimate_document_type(document_content: str, llm_client: Any) -> str:
    """Estimate the document type from document content using LLM."""
    # Take the first 2000 characters for analysis to avoid token limits
    content_sample = document_content[:2000]
    prompt = f"""
    Analyze the following legal document excerpt and determine the document type.
    Provide only the document type (e.g., "contract", "agreement", "policy", "statute").
    
    Document excerpt:
    {content_sample}
    """
    
    try:
        response = llm_client.query(prompt)
        # Clean the response to get just the document type
        document_type = response.strip().split('\n')[0].strip()
        return document_type or "contract"  # Default to contract if empty
    except Exception as e:
        print(f"ERROR: Error estimating document type: {str(e)}")
        return "contract"  # Default to contract on error


def _prepare_knowledge_context(knowledge_from_vector_db: List[Dict[str, Any]], llm_client: Any) -> Dict[str, Any]:
    """
    Prepare and categorize knowledge context from vector database results.
    
    Args:
        knowledge_from_vector_db: Raw results from vector database
        llm_client: LLM client for classification
        
    Returns:
        Dict: Structured knowledge context with statutes and precedents
    """
    knowledge_context = {
        "statutes": [],
        "precedents": []
    }
    
    if not knowledge_from_vector_db:
        print("WARNING: No knowledge data from vector database")
        return knowledge_context
    
    print(f"Processing {len(knowledge_from_vector_db)} knowledge items from vector database")
    
    # Process each knowledge item
    for item in knowledge_from_vector_db:
        content = item.get("content", "")
        title = item.get("title", "")
        
        if not content and not title:
            print("WARNING: Skipping empty knowledge item")
            continue
        
        # Classify the item as statute or precedent
        item_type = _classify_knowledge_item(title, content, llm_client)
        
        if item_type == "statute":
            knowledge_context["statutes"].append({
                "title": title,
                "content": content,
                "url": item.get("url", ""),
                "score": item.get("score", 0.0)
            })
        else:  # precedent
            knowledge_context["precedents"].append({
                "title": title,
                "content": content,
                "url": item.get("url", ""),
                "score": item.get("score", 0.0)
            })
    
    print(f"Classified knowledge: {len(knowledge_context['statutes'])} statutes, {len(knowledge_context['precedents'])} precedents")
    return knowledge_context


def _classify_knowledge_item(title: str, content: str, llm_client: Any) -> str:
    """
    Classify a knowledge item as either a statute or precedent using LLM.
    
    Args:
        title: Item title
        content: Item content
        llm_client: LLM client for classification
        
    Returns:
        str: Either "statute" or "precedent"
    """
    # Take a sample of the content to avoid token limits
    content_sample = content[:500] if content else ""
    
    prompt = f"""
    Determine if the following legal text is more likely to be a statute/regulation or a legal precedent/case law.
    
    Title: {title}
    Content excerpt: {content_sample}
    
    Respond with only one word: either "statute" or "precedent".
    """
    
    try:
        response = llm_client.query(prompt).strip().lower()
        if "statute" in response or "regulation" in response or "code" in response:
            return "statute"
        else:
            return "precedent"
    except Exception as e:
        print(f"ERROR: Error classifying knowledge item: {str(e)}")
        # Default based on title keywords if LLM fails
        statute_keywords = ["code", "statute", "act", "regulation", "rule", "law"]
        if any(keyword in title.lower() for keyword in statute_keywords):
            return "statute"
        return "precedent"


def _analyze_clause_compliance(
    clause: Dict[str, Any],
    context_bank: ContextBank,
    all_clauses: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    jurisdiction: str,
    document_type: str,
    knowledge_context: Dict[str, Any],
    llm_client: Any,
    min_confidence: float
) -> Dict[str, Any]:
    """
    Analyze a single clause for compliance issues.
    
    Args:
        clause: The clause to analyze
        context_bank: The context bank
        all_clauses: All clauses in the document
        entities: All entities in the document
        jurisdiction: Document jurisdiction
        document_type: Document type
        knowledge_context: Structured knowledge context
        llm_client: LLM client
        min_confidence: Minimum confidence threshold
        
    Returns:
        Dict: Comprehensive analysis results
    """
    clause_id = clause.get("id", str(uuid.uuid4()))
    clause_text = clause.get("Text", "")
    
    if not clause_text:
        print(f"WARNING: Empty text for clause ID {clause_id}")
        return {
            "has_issues": False,
            "issue_count": 0,
            "result": None
        }
    
    print(f"Analyzing compliance for clause ID {clause_id}")
    
    # 1. Check statutory compliance - only if statutes are available
    statutory_violations = []
    if knowledge_context["statutes"]:
        print(f"Checking statutory compliance against {len(knowledge_context['statutes'])} statutes")
        try:
            # Adding a delay here to avoid hitting API rate limits
            time.sleep(5)

            statutory_violations = validate_statutory_compliance(
                clause_text=clause_text,
                llm_client=llm_client,
                clause_id=clause_id,
                jurisdiction=jurisdiction,
                knowledge_context=knowledge_context["statutes"],
                min_confidence=min_confidence
            )
            print(f"Found {len(statutory_violations)} statutory violations")
        except Exception as e:
            print(f"ERROR: Error in statutory validation: {str(e)}")
    
    # 2. Check precedent compliance - only if precedents are available
    precedent_issues = []
    if knowledge_context["precedents"]:
        print(f"Checking precedent compliance against {len(knowledge_context['precedents'])} precedents")
        try:
            # Adding a delay here to avoid hitting API rate limits
            time.sleep(5)

            # Call the implemented function
            precedent_issues = analyze_precedents_for_compliance(
                clause_text=clause_text,
                llm_client=llm_client,
                clause_id=clause_id, # Pass clause_id
                jurisdiction=jurisdiction,
                document_type=document_type, # Pass document_type
                knowledge_context=knowledge_context["precedents"], # Pass only precedents
                min_confidence=min_confidence
            )
            print(f"Found {len(precedent_issues)} precedent issues")
        except Exception as e:
            print(f"ERROR: Error in precedent analysis: {str(e)}")
    
    # 3. Check contractual consistency
    consistency_issues = []
    try:
        # Prepare clauses for consistency check
        consistency_clauses = [{"id": clause_id, "text": clause_text}]
        
        # Add other clauses (excluding current one)
        for other_clause in all_clauses:
            other_clause_id = other_clause.get("id")
            if other_clause_id and other_clause_id != clause_id:
                consistency_clauses.append({
                    "id": other_clause_id,
                    "text": other_clause.get("text", "")
                })
        
        print(f"Checking consistency against {len(consistency_clauses)-1} other clauses")
        
        # Document context for consistency check
        document_context = {
            "jurisdiction": jurisdiction,
            "document_type": document_type,
            "entities": entities
        }
        
        # Adding a delay here to avoid hitting API rate limits
        time.sleep(5)

        # Perform consistency check
        consistency_analysis = check_contractual_consistency(
            clauses=consistency_clauses,
            llm_client=llm_client,
            document_context=document_context,
            min_confidence=min_confidence,
            use_hypergraph=True
        )
        
        # Extract issues relating to the current clause
        for inconsistency in consistency_analysis.get("inconsistencies", []):
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
                    "confidence": inconsistency.confidence,
                    "type": "consistency"
                })
        
        print(f"Found {len(consistency_issues)} consistency issues")
    except Exception as e:
        print(f"ERROR: Error in consistency analysis: {str(e)}")
    
    # 4. Optional: Perform hypergraph analysis if there are enough clauses
    hypergraph_analysis = None
    if len(consistency_clauses) >= 3:
        try:
            print("Performing hypergraph analysis")
            hypergraph_analysis = analyze_hypergraph_structure(
                clauses=consistency_clauses,
                llm_client=llm_client,
                analyze_cycles=True,
                analyze_critical_nodes=True,
                analyze_clusters=True,
                node_to_analyze=clause_id
            )
            print("Completed hypergraph analysis")
        except Exception as e:
            print(f"ERROR: Error in hypergraph analysis: {str(e)}")
    
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
    
    # Add precedent issues (now directly a list of dicts)
    precedent_issues_count = len(precedent_issues) # Get count before extending
    clause_issues.extend(precedent_issues)

    # Add consistency issues (already has type set)
    clause_issues.extend(consistency_issues)
    
    # Prepare the result for non-compliant clauses
    if clause_issues:
        non_compliant_clause = {
            "clause_id": clause_id,
            "clause_text": clause_text,
            "issues": clause_issues,
            "issue_count": len(clause_issues),
            "statutory_violations": len(statutory_violations),
            "precedent_issues": precedent_issues_count, # Use the count variable
            "consistency_issues": len(consistency_issues)
        }

        # Add hypergraph analysis if available
        if hypergraph_analysis:
            non_compliant_clause["hypergraph_analysis"] = {
                "cycles": len(hypergraph_analysis.get("cycles", [])),
                "critical_nodes": len(hypergraph_analysis.get("critical_nodes", [])),
                "has_impact_analysis": hypergraph_analysis.get("impact_analysis") is not None,
                "relationship_clusters": len(hypergraph_analysis.get("relationship_clusters", []))
            }

        # Store in context bank
        clause_analysis = {
            "clause_id": clause_id,
            "clause_text": clause_text,
            "statutory_violations": [v.__dict__ for v in statutory_violations],
            "precedent_issues": precedent_issues, # Store the actual issues list
            "consistency_issues": consistency_issues,
            "hypergraph_analysis": hypergraph_analysis,
            "all_issues": clause_issues,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            context_bank.add_clause_compliance_result(clause_id=clause_id, analysis=clause_analysis)
            print(f"Stored compliance analysis for clause {clause_id} in context bank")
        except Exception as e:
            print(f"ERROR: Error storing clause analysis in context bank: {str(e)}")

        return {
            "has_issues": True,
            "issue_count": len(clause_issues),
            "result": non_compliant_clause
        }
    else:
        return {
            "has_issues": False,
            "issue_count": 0,
            "result": None
        }


def _store_document_analysis(
    context_bank: ContextBank,
    clauses_analyzed: int,
    non_compliant_clauses: List[Dict[str, Any]],
    jurisdiction: str,
    document_type: str
) -> None:
    """
    Store document-level analysis results in context bank.
    
    Args:
        context_bank: The context bank
        clauses_analyzed: Number of clauses analyzed
        non_compliant_clauses: List of non-compliant clauses
        jurisdiction: Document jurisdiction
        document_type: Document type
    """
    total_issues = sum(clause.get("issue_count", 0) for clause in non_compliant_clauses)
    
    document_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "clauses_analyzed": clauses_analyzed,
        "non_compliant_clauses": len(non_compliant_clauses),
        "total_issues": total_issues,
        "has_issues": len(non_compliant_clauses) > 0,
        "timestamp": datetime.now().isoformat(),
        "jurisdiction": jurisdiction,
        "document_type": document_type
    }
    
    try:
        context_bank.add_document_analysis(analysis=document_analysis)
        print(f"Stored document analysis in context bank: {len(non_compliant_clauses)} non-compliant clauses with {total_issues} issues")
    except Exception as e:
        print(f"ERROR: Error storing document analysis in context bank: {str(e)}")
