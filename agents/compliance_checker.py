# agents/compliance_checker.py
from typing import Dict, List, Any, Optional
import uuid
from context_bank import ContextBank

from utils.ollama_client import OllamaClient
from utils.api_client import APIClient
from utils.prompt_templates import PromptTemplates
from utils.response_parser import ResponseParser
from utils.confidence_scorer import ConfidenceScorer

from agents.statutory_validator import StatutoryValidator, Violation, SeverityLevel
from agents.precedent_analyzer import PrecedentAnalyzer
from agents.consistency_engine import ContractualConsistencyEngine
from agents.hypergraph_analyzer import HypergraphAnalyzer
from agents.knowledge import KnowledgeAgent

class ComplianceCheckerAgent:
    """
    Compliance Checker Agent that detects contradictions between clauses
    and with relevant laws, provides reasoning, and extrapolates legal implications.
    
    This agent leverages the context bank for shared memory and the knowledge agent 
    for retrieving relevant legal information.
    """
    
    def __init__(self, use_ollama: bool, model_name: str, context_bank: ContextBank, 
                 min_confidence: float = 0.75, knowledge_agent=None):
        """
        Initialize the Compliance Checker Agent.
        
        Args:
            use_ollama: Whether to use Ollama for local model inference
            model_name: Name of the model to use
            context_bank: Shared context bank for all agents
            min_confidence: Minimum confidence threshold for valid issues
            knowledge_agent: Optional knowledge agent for statute retrieval
        """
        # Store references to shared components
        self.context_bank = context_bank
        self.knowledge_agent = knowledge_agent
        
        # Initialize the appropriate client based on configuration
        if use_ollama:
            self.llm_client = OllamaClient(model_name)
        else:
            self.llm_client = APIClient(model_name)
        
        # Initialize utility components
        self.prompt_templates = PromptTemplates()
        self.response_parser = ResponseParser()
        self.confidence_scorer = ConfidenceScorer(min_confidence_threshold=min_confidence)
        
        # Initialize specialized validators and analyzers
        self.statutory_validator = StatutoryValidator(
            llm_client=self.llm_client,
            knowledge_agent=knowledge_agent,  # Pass the knowledge agent for statute retrieval
            min_confidence=min_confidence
        )
        
        # Initialize the precedent analyzer
        self.precedent_analyzer = PrecedentAnalyzer(
            llm_client=self.llm_client,
            min_confidence=min_confidence,
            use_web_search=False  # Set to True if web search is available
        )
        
        # Initialize the consistency engine
        self.consistency_engine = ContractualConsistencyEngine(
            llm_client=self.llm_client,
            min_confidence=min_confidence,
            use_hypergraph=True  # Enable hypergraph analysis for complex relationships
        )
        
        # Initialize the hypergraph analyzer
        self.hypergraph_analyzer = HypergraphAnalyzer(
            llm_client=self.llm_client
        )
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document state to check for compliance issues.
        This is the main entry point for the orchestrator.
        
        Args:
            state: Current state of the workflow, including document_id
            
        Returns:
            Dict: Updated state with compliance analysis results
        """
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
            
        # Prepare results container
        compliance_results = {
            "document_id": document_id,
            "analysis_id": str(uuid.uuid4()),
            "clauses_analyzed": len(clauses),
            "total_issues": 0,
            "clause_results": [],
            "has_contradictions": False
        }
        
        # Process each clause for compliance issues
        for clause in clauses:
            # Retrieve relevant context for this clause
            jurisdiction = document.get("metadata", {}).get("jurisdiction", "US")
            document_context = {
                "jurisdiction": jurisdiction,
                "clauses": [c for c in clauses if c["id"] != clause["id"]],
                "document_type": document.get("metadata", {}).get("document_type", "contract")
            }
            
            # Retrieve any relevant statutes and precedents from knowledge agent
            if self.knowledge_agent:
                # Get relevant laws for this clause from the knowledge agent
                # This is handled internally by the statutory_validator

                # Add any additional context from knowledge agent
                knowledge_context = self._get_knowledge_context(clause, jurisdiction)
                if knowledge_context:
                    document_context["knowledge_context"] = knowledge_context
            
            # Perform comprehensive compliance check
            result = self.check_compliance(
                clause_text=clause["text"],
                clause_id=clause["id"],
                document_context=document_context
            )
            
            # Store results for this clause
            compliance_results["clause_results"].append(result)
            compliance_results["total_issues"] += result["issue_count"]
            
            # Check if any contradictions were found
            if result["has_issues"]:
                compliance_results["has_contradictions"] = True
        
        # Store complete analysis in context bank
        self.context_bank.store(
            key=f"document_compliance_analysis:{document_id}",
            value=compliance_results
        )
        
        # If analyzing a complete document with multiple clauses,
        # perform a document-level structure analysis
        if len(clauses) >= 3:
            structure_analysis = self.analyze_document_structure(clauses)
            self.context_bank.store(
                key=f"document_structure_analysis:{document_id}",
                value=structure_analysis
            )
            compliance_results["structure_analysis"] = structure_analysis
        
        # Update state for next agent
        state["compliance_analyzed"] = True
        state["compliance_analysis_id"] = compliance_results["analysis_id"]
        state["has_contradictions"] = compliance_results["has_contradictions"]
        
        # Determine next step based on whether contradictions were found
        if compliance_results["has_contradictions"]:
            state["next_step"] = "clause_rewriter"  # Send to rewriter if there are issues
        else:
            state["next_step"] = "post_processor"   # Send to post-processor if no issues
            
        return state
    
    def _get_knowledge_context(self, clause: Dict[str, Any], jurisdiction: str) -> Dict[str, Any]:
        """
        Retrieve relevant knowledge context for a clause.
        
        Args:
            clause: Clause to get context for
            jurisdiction: Legal jurisdiction
            
        Returns:
            Dict: Knowledge context for the clause
        """
        if not self.knowledge_agent:
            return {}
            
        # In a real implementation, you'd call specific methods on the knowledge agent
        # Here we're assuming the knowledge is already stored in the context bank
        # by the knowledge agent's earlier processing
        
        # Construct a key for the context bank lookup
        knowledge_key = f"knowledge:{clause['id']}"
        
        # Try to retrieve existing knowledge
        knowledge = self.context_bank.get(knowledge_key, None)
        
        if knowledge:
            return knowledge
            
        # If no existing knowledge, try to retrieve it directly
        # This is a fallback mechanism
        try:
            # Query knowledge agent for relevant statutes
            statutes = self.knowledge_agent.find_relevant_statutes(
                query=clause["text"],
                jurisdiction=jurisdiction
            )
            
            # Query knowledge agent for relevant precedents
            precedents = self.knowledge_agent.find_relevant_precedents(
                query=clause["text"],
                jurisdiction=jurisdiction
            )
            
            # Compile the knowledge context
            knowledge_context = {
                "statutes": statutes,
                "precedents": precedents
            }
            
            # Store for future use
            self.context_bank.store(key=knowledge_key, value=knowledge_context)
            
            return knowledge_context
            
        except Exception as e:
            print(f"Error retrieving knowledge context: {e}")
            return {}
    
    def check_compliance(self, clause_text: str, clause_id: str = None, 
                        document_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform comprehensive compliance checking on a clause.
        
        Args:
            clause_text: Text of the clause to check
            clause_id: Optional identifier for the clause
            document_context: Optional context from the broader document
            
        Returns:
            Dict: Compliance analysis results
        """
        if clause_id is None:
            clause_id = str(uuid.uuid4())
        
        # Prepare the document context
        jurisdiction = document_context.get("jurisdiction", "US") if document_context else "US"
        other_clauses = document_context.get("clauses", []) if document_context else []
        
        # Retrieve knowledge context if available
        knowledge_context = document_context.get("knowledge_context", {}) if document_context else {}
        
        # Run statutory validation using the dedicated validator
        # The StatutoryValidator will use the knowledge agent if available
        statutory_violations = self.statutory_validator.validate_clause(
            clause_text=clause_text,
            clause_id=clause_id,
            jurisdiction=jurisdiction
        )
        
        # Run precedent analysis using the dedicated analyzer
        # Pass relevant precedents from knowledge context if available
        precedent_context = knowledge_context.get("precedents", [])
        precedent_analysis = self.precedent_analyzer.analyze_precedents(
            clause_text=clause_text,
            jurisdiction=jurisdiction,
            context={
                "precedents": precedent_context,
                **document_context
            } if precedent_context else document_context
        )
        precedent_issues = precedent_analysis.get("issues", [])
        
        # Run consistency check using the dedicated engine
        # First, prepare the clauses list
        current_clause = {"id": clause_id, "text": clause_text}
        all_clauses = [current_clause]
        
        if other_clauses:
            for other_clause in other_clauses:
                if isinstance(other_clause, dict) and "text" in other_clause:
                    # Make sure each clause has an ID
                    if "id" not in other_clause:
                        other_clause["id"] = str(uuid.uuid4())
                    all_clauses.append(other_clause)
        
        # Run the consistency check
        consistency_analysis = self.consistency_engine.check_consistency(
            clauses=all_clauses,
            document_context=document_context
        )
        
        # Run hypergraph analysis if there are enough clauses
        # This provides deeper insights into document structure
        hypergraph_analysis = None
        if len(all_clauses) >= 3:
            # Build the hypergraph
            legal_graph = self.hypergraph_analyzer.build_graph(all_clauses)
            
            # Detect cycles
            cycles = self.hypergraph_analyzer.detect_cycles(legal_graph)
            
            # Find critical nodes
            critical_nodes = self.hypergraph_analyzer.find_critical_nodes(legal_graph)
            
            # Analyze impact of the current clause
            current_node_id = None
            for node_id, node in legal_graph.nodes.items():
                if node.data.get("id") == clause_id:
                    current_node_id = node_id
                    break
            
            impact_analysis = None
            if current_node_id:
                impact_analysis = self.hypergraph_analyzer.analyze_impact(current_node_id, legal_graph)
            
            # Analyze relationship clusters
            clusters = self.hypergraph_analyzer.analyze_relationship_clusters(legal_graph)
            
            # Compile the hypergraph analysis results
            hypergraph_analysis = {
                "cycles": [cycle.__dict__ for cycle in cycles],
                "critical_nodes": critical_nodes,
                "impact_analysis": impact_analysis.__dict__ if impact_analysis else None,
                "relationship_clusters": clusters
            }
        
        # Convert Inconsistency objects to dictionaries
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
        
        # Combine all results
        all_issues = []
        
        # Add statutory violations
        for violation in statutory_violations:
            all_issues.append({
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
            all_issues.append(issue)
            
        # Add consistency issues
        for issue in consistency_issues:
            issue["type"] = "consistency"
            all_issues.append(issue)
        
        # Also analyze business implications
        business_context = document_context.get("business_context", {}) if document_context else {}
        implications = self.analyze_implications(clause_text, business_context)
        
        # Store results in context bank for use by other agents
        self.context_bank.store(
            key=f"compliance_analysis:{clause_id}",
            value={
                "clause_id": clause_id,
                "clause_text": clause_text,
                "statutory_violations": [v.__dict__ for v in statutory_violations],
                "precedent_issues": precedent_issues,
                "precedent_analysis": precedent_analysis,
                "consistency_issues": consistency_issues,
                "consistency_analysis": consistency_analysis,
                "hypergraph_analysis": hypergraph_analysis,  # Store the hypergraph analysis
                "implications": implications,  # Store business implications
                "all_issues": all_issues,
                "timestamp": datetime.now().isoformat()  # Add timestamp for tracking
            }
        )
        
        # Record any contradictions in the context bank's dedicated storage
        if all_issues:
            for issue in all_issues:
                self.context_bank.add_contradiction(
                    document_id=document_context.get("document_id", "unknown"),
                    contradiction={
                        "clause_id": clause_id,
                        "issue_type": issue["type"],
                        "description": issue["description"],
                        "severity": issue["severity"],
                        "references": issue.get("references", []),
                        "implications": issue.get("implications", [])
                    }
                )
        
        # Add hypergraph analysis to the return value
        result = {
            "clause_id": clause_id,
            "has_issues": len(all_issues) > 0,
            "issue_count": len(all_issues),
            "issues": all_issues,
            "statutory_violations": len(statutory_violations),
            "precedent_issues": len(precedent_issues),
            "consistency_issues": len(consistency_issues),
            "implications": len(implications)
        }
        
        if hypergraph_analysis:
            result["hypergraph_analysis"] = {
                "cycles": len(hypergraph_analysis["cycles"]),
                "critical_nodes": len(hypergraph_analysis["critical_nodes"]),
                "has_impact_analysis": hypergraph_analysis["impact_analysis"] is not None,
                "relationship_clusters": len(hypergraph_analysis["relationship_clusters"])
            }
        
        return result
    
    def analyze_implications(self, clause_text: str, business_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Analyze potential implications of a clause.
        
        Args:
            clause_text: Text of the clause to analyze
            business_context: Optional business context information
            
        Returns:
            List: Potential implications
        """
        # Format the prompt using the template utility
        user_prompt = self.prompt_templates.format_implication_prompt(
            clause_text=clause_text,
            context=business_context
        )
        
        # Get implication analysis from LLM
        response = self.llm_client.generate(
            system_prompt=self.prompt_templates.implication_analysis_template,
            user_prompt=user_prompt
        )
        
        # Parse the response to extract issues (which represent implications in this context)
        analysis = self.response_parser.parse_issues(response)
        
        # Format the results
        implications = []
        for issue in analysis:
            implications.append({
                "description": issue.get("description", ""),
                "category": self._categorize_implication(issue.get("description", "")),
                "severity": issue.get("severity", "MEDIUM"),
                "details": issue.get("reasoning", ""),
                "recommendations": self._extract_recommendations(issue.get("reasoning", ""))
            })
            
        return implications
    
    def _categorize_implication(self, description: str) -> str:
        """
        Categorize an implication based on its description.
        
        Args:
            description: Implication description
            
        Returns:
            str: Category (legal, financial, operational, reputational)
        """
        description = description.lower()
        
        if any(term in description for term in ["law", "legal", "statute", "regulation", "compliance", "liability"]):
            return "legal"
        elif any(term in description for term in ["cost", "expense", "financial", "monetary", "budget", "revenue"]):
            return "financial"
        elif any(term in description for term in ["process", "operation", "workflow", "efficiency", "implementation"]):
            return "operational"
        elif any(term in description for term in ["reputation", "brand", "public", "perception", "image"]):
            return "reputational"
        else:
            return "general"
    
    def _extract_recommendations(self, reasoning: str) -> List[str]:
        """
        Extract recommendations from reasoning text.
        
        Args:
            reasoning: Reasoning text
            
        Returns:
            List[str]: Extracted recommendations
        """
        recommendations = []
        
        # Look for recommendation patterns in the text
        lower_text = reasoning.lower()
        
        # Check for specific recommendation markers
        markers = ["recommend", "suggest", "should", "could", "consider", "advisable", "proposed solution"]
        
        sentences = reasoning.split(". ")
        for sentence in sentences:
            if any(marker in sentence.lower() for marker in markers):
                recommendations.append(sentence.strip())
                
        return recommendations
    
    # Add a new method for document-level hypergraph analysis
    def analyze_document_structure(self, clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform document-level structural analysis using hypergraphs.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            Dict: Document structure analysis results
        """
        if not clauses or len(clauses) < 2:
            return {
                "error": "Not enough clauses for meaningful analysis",
                "clause_count": len(clauses) if clauses else 0
            }
        
        # Build the hypergraph
        legal_graph = self.hypergraph_analyzer.build_graph(clauses)
        
        # Detect cycles
        cycles = self.hypergraph_analyzer.detect_cycles(legal_graph)
        
        # Find critical nodes
        critical_nodes = self.hypergraph_analyzer.find_critical_nodes(legal_graph)
        
        # Analyze relationship clusters
        clusters = self.hypergraph_analyzer.analyze_relationship_clusters(legal_graph)
        
        # Compile the analysis results
        analysis = {
            "document_id": str(uuid.uuid4()),
            "clause_count": len(clauses),
            "cycles": [cycle.__dict__ for cycle in cycles],
            "cycle_count": len(cycles),
            "critical_nodes": critical_nodes,
            "critical_node_count": len(critical_nodes),
            "relationship_clusters": clusters,
            "cluster_count": len(clusters),
            "has_structural_issues": len(cycles) > 0 or any(node["criticality_score"] > 10 for node in critical_nodes)
        }
        
        # Store the analysis in the context bank
        self.context_bank.store(
            key=f"document_structure_analysis:{analysis['document_id']}",
            value=analysis
        )
        
        return analysis
