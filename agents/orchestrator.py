# agents/orchestrator.py
from typing import Dict, List, Any
from context_bank import ContextBank
from utils.ollama_client import OllamaClient
from utils.api_client import APIClient

class OrchestratorAgent:
    """
    Orchestrator Agent that manages the workflow and task planning for the legal document
    review process. It creates or updates a ledger of tasks, analyzes clauses, looks up
    legal clauses, makes educated guesses, and creates a task plan.
    """
    
    def __init__(self, use_ollama: bool, model_name: str, context_bank: ContextBank):
        """
        Initialize the Orchestrator Agent.
        
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
        
        # System prompt for the task planner
        self.task_planner_prompt = """
        You are a Task Planner for legal document review. Your job is to:
        1. Create or update a ledger of tasks
        2. Analyze given clauses
        3. Identify legal clauses to look up
        4. Make educated guesses about potential issues
        5. Create a comprehensive task plan
        
        Based on the current state of document processing, determine the next steps needed.
        """
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and determine the next steps in the workflow.
        
        Args:
            state: Current state of the workflow
            
        Returns:
            Dict: Updated state with next steps
        """
        # Extract relevant information from the state
        document_id = state.get("document_id")
        document = self.context_bank.get_document(document_id)
        task_ledger = state.get("task_ledger", [])
        
        # If this is a new document that hasn't been processed yet
        if document and not document.get("processed", False):
            # Set next step to preprocess the document
            state["next_step"] = "preprocess"
            task_ledger.append({
                "task": "preprocess_document",
                "status": "pending",
                "document_id": document_id
            })
            state["task_ledger"] = task_ledger
            return state
        
        # If we have clauses but haven't checked for contradictions
        if document_id in self.context_bank.clauses and document_id not in self.context_bank.contradictions:
            # Set next step to check compliance
            state["next_step"] = "compliance"
            task_ledger.append({
                "task": "check_compliance",
                "status": "pending",
                "document_id": document_id
            })
            state["task_ledger"] = task_ledger
            return state
        
        # If we have contradictions but no suggestions
        contradictions = self.context_bank.get_all_contradictions(document_id)
        if contradictions and not self.context_bank.get_all_suggestions(document_id):
            # Set next step to rewrite clauses
            state["next_step"] = "rewrite"
            task_ledger.append({
                "task": "rewrite_clauses",
                "status": "pending",
                "document_id": document_id,
                "contradiction_ids": [c.get("id") for c in contradictions]
            })
            state["task_ledger"] = task_ledger
            return state
        
        # If we have suggestions but haven't generated a final report
        if self.context_bank.get_all_suggestions(document_id) and not state.get("report_generated"):
            # Set next step to postprocess
            state["next_step"] = "postprocess"
            task_ledger.append({
                "task": "generate_report",
                "status": "pending",
                "document_id": document_id
            })
            state["task_ledger"] = task_ledger
            return state
        
        # If all tasks are complete
        state["next_step"] = "complete"
        state["complete"] = True
        return state
    
    def _plan_tasks(self, document_id: str, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a task plan based on the current state of processing.
        
        Args:
            document_id: ID of the document being processed
            current_state: Current state of processing
            
        Returns:
            List: Planned tasks
        """
        # Prepare the input for the LLM
        document = self.context_bank.get_document(document_id)
        clauses = self.context_bank.clauses.get(document_id, [])
        
        input_text = f"""
        Document ID: {document_id}
        Document Type: {document.get('metadata', {}).get('type', 'Unknown')}
        Number of Clauses: {len(clauses)}
        Current Processing Stage: {current_state.get('current_stage', 'New Document')}
        
        Based on this information, create a task plan for processing this document.
        """
        
        # Get task plan from LLM
        response = self.llm_client.generate(
            system_prompt=self.task_planner_prompt,
            user_prompt=input_text
        )
        
        # Parse the response to extract tasks
        # This is a simplified implementation - in a real system, you'd want more robust parsing
        tasks = []
        for line in response.strip().split('\n'):
            if line.startswith('- '):
                task_desc = line[2:].strip()
                tasks.append({
                    "description": task_desc,
                    "status": "pending"
                })
        
        return tasks
