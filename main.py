# main.py
import os
from typing import Dict, List, Any
from dotenv import load_dotenv
import langgraph.graph as lg
from langgraph.graph import StateGraph, END

# Import agent modules
from agents.orchestrator import OrchestratorAgent
from agents.preprocessor import PreprocessorAgent
from agents.knowledge import KnowledgeAgent
from agents.compliance_checker import ComplianceCheckerAgent
from agents.clause_rewriter import ClauseRewriterAgent
from agents.postprocessor import PostprocessorAgent
from context_bank import ContextBank

# Load environment variables from a .env file
load_dotenv()

class LegalDiscrepancyDetectionFramework:
    """
    Main framework class that initializes and connects all agents in the legal
    discrepancy detection system using LangGraph.
    """
    
    def __init__(self, use_ollama: bool = True, model_name: str = "llama3"):
        """
        Initialize the framework with configuration for model usage.
        
        Args:
            use_ollama: Whether to use Ollama for local model inference (True) or external APIs (False)
            model_name: Name of the model to use with Ollama
        """
        self.use_ollama = use_ollama
        self.model_name = model_name
        self.context_bank = ContextBank()
        
        # Initialize all agents with shared context bank
        self.orchestrator = OrchestratorAgent(use_ollama, model_name, self.context_bank)
        self.preprocessor = PreprocessorAgent(use_ollama, model_name, self.context_bank)
        self.knowledge_agent = KnowledgeAgent(use_ollama, model_name, self.context_bank)
        self.compliance_checker = ComplianceCheckerAgent(use_ollama, model_name, self.context_bank)
        self.clause_rewriter = ClauseRewriterAgent(use_ollama, model_name, self.context_bank)
        self.postprocessor = PostprocessorAgent(use_ollama, model_name, self.context_bank)
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow that defines how agents interact.
        
        Returns:
            StateGraph: The configured workflow graph
        """
        # Define the state schema
        workflow = StateGraph(name="legal_document_review")
        
        # Add nodes for each agent's processing function
        workflow.add_node("orchestrator", self.orchestrator.process)
        workflow.add_node("preprocessor", self.preprocessor.process)
        workflow.add_node("knowledge_agent", self.knowledge_agent.process)
        workflow.add_node("compliance_checker", self.compliance_checker.process)
        workflow.add_node("clause_rewriter", self.clause_rewriter.process)
        workflow.add_node("postprocessor", self.postprocessor.process)
        
        # Define the edges (workflow connections)
        # Start with orchestrator
        workflow.set_entry_point("orchestrator")
        
        # Define conditional routing based on orchestrator's decision
        workflow.add_conditional_edges(
            "orchestrator",
            self._route_from_orchestrator,
            {
                "preprocess": "preprocessor",
                "knowledge": "knowledge_agent",
                "compliance": "compliance_checker",
                "rewrite": "clause_rewriter",
                "postprocess": "postprocessor",
                "complete": END
            }
        )
        
        # From preprocessor to orchestrator for next steps
        workflow.add_edge("preprocessor", "orchestrator")
        
        # From knowledge agent back to orchestrator or to compliance checker
        workflow.add_conditional_edges(
            "knowledge_agent",
            self._route_from_knowledge,
            {
                "orchestrator": "orchestrator",
                "compliance": "compliance_checker"
            }
        )
        
        # From compliance checker to clause rewriter or back to orchestrator
        workflow.add_conditional_edges(
            "compliance_checker",
            self._route_from_compliance,
            {
                "rewrite": "clause_rewriter",
                "orchestrator": "orchestrator"
            }
        )
        
        # From clause rewriter to compliance checker to verify fixes or to orchestrator
        workflow.add_conditional_edges(
            "clause_rewriter",
            self._route_from_rewriter,
            {
                "compliance": "compliance_checker",
                "orchestrator": "orchestrator"
            }
        )
        
        # From postprocessor to end or back to orchestrator
        workflow.add_conditional_edges(
            "postprocessor",
            self._route_from_postprocessor,
            {
                "complete": END,
                "orchestrator": "orchestrator"
            }
        )
        
        return workflow.compile()
    
    def _route_from_orchestrator(self, state: Dict[str, Any]) -> str:
        """
        Determine the next step based on orchestrator output.
        
        Args:
            state: Current state of the workflow
            
        Returns:
            str: Next node to route to
        """
        return state.get("next_step", "complete")
    
    def _route_from_knowledge(self, state: Dict[str, Any]) -> str:
        """
        Determine where to route after knowledge agent processing.
        
        Args:
            state: Current state of the workflow
            
        Returns:
            str: Next node to route to
        """
        return state.get("next_step", "orchestrator")
    
    def _route_from_compliance(self, state: Dict[str, Any]) -> str:
        """
        Determine where to route after compliance checking.
        
        Args:
            state: Current state of the workflow
            
        Returns:
            str: Next node to route to
        """
        # If contradictions found, go to rewriter, else back to orchestrator
        if state.get("contradictions_found", False):
            return "rewrite"
        return "orchestrator"
    
    def _route_from_rewriter(self, state: Dict[str, Any]) -> str:
        """
        Determine where to route after clause rewriting.
        
        Args:
            state: Current state of the workflow
            
        Returns:
            str: Next node to route to
        """
        # If rewrite needs verification, go back to compliance, else to orchestrator
        if state.get("verify_rewrite", False):
            return "compliance"
        return "orchestrator"
    
    def _route_from_postprocessor(self, state: Dict[str, Any]) -> str:
        """
        Determine where to route after postprocessing.
        
        Args:
            state: Current state of the workflow
            
        Returns:
            str: Next node to route to
        """
        if state.get("complete", True):
            return "complete"
        return "orchestrator"
    
    def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process a legal document through the entire framework.
        
        Args:
            document_path: Path to the document file
            
        Returns:
            Dict: Results of the document analysis
        """
        # Initialize the state with the document path
        initial_state = {
            "document_path": document_path,
            "document_id": os.path.basename(document_path),
            "next_step": "preprocess",
            "task_ledger": [],
            "contradictions": [],
            "suggestions": [],
            "complete": False
        }
        
        # Execute the workflow
        result = self.workflow.invoke(initial_state)
        return result

if __name__ == "__main__":
    # Example usage
    framework = LegalDiscrepancyDetectionFramework(use_ollama=True, model_name="llama3")
    result = framework.process_document("path/to/legal_document.pdf")
    print("Analysis complete. Results:", result)
