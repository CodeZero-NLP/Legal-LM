# agents/orchestrator.py
from typing import Dict, List, Any
from context_bank import ContextBank
from utils.ollama_client import OllamaClient
from utils.api_client import APIClient

from utils.gemini_client import GeminiClient 

class OrchestratorAgent:
    """
    Orchestrator Agent that manages the workflow and task planning for the legal document
    review process. It creates or updates a ledger of tasks, analyzes clauses, looks up
    legal clauses, makes educated guesses, and creates a task plan.
    """

    def __init__(self, use_ollama: bool, model_name: str, context_bank: ContextBank, use_gemini: bool = False):
        self.context_bank = context_bank
        if use_gemini:
            self.llm_client = GeminiClient(model_name)
        elif use_ollama:
            self.llm_client = OllamaClient(model_name)
        else:
            self.llm_client = APIClient(model_name)

        self.task_planner_prompt = """
        You are a Task Planner Agent responsible for coordinating a multi-agent system to analyze legal documents for discrepancies and compliance. Your job is to plan and delegate tasks to specialized agents, track task completion, and dynamically adapt the plan based on the current system state.

        You are aware of the capabilities of the agents and can query or instruct them based on the task at hand. After every task execution, you should validate whether the task was completed successfully. If a task fails or the output is insufficient, you should modify the workflow, reassign the task, or create additional subtasks.

        INPUTS: 
        Problem to solve:
        {Insert Problem Statement i.e. User Prompt}

        Planned tasks:
        - Preprocess document
        - Extract and classify clauses
        - Check compliance
        - Rewrite non-compliant clauses
        - Summarize issues

        Tasks done:
        {Framework Status / Task Logs}

        Here are the Agents you have access to:
        Preprocessor Agent: Parses documents, title of document, extracts clauses, extracts named entities and classifies the document.
        Knowledge Agent: Retrieves Knowledge from Web. Input should be specific on what information is to be retrieved.
        Clause Compliance Checker Agent: Checks the compliance of all the clauses by accessing the knowledge agent. If it is non-compliant, it will flag those clauses while also mentioning the issue in it. It will ask the Clause Rewriter agent to do it.
        Post Processor agent: If the document is compliant, then return the issues in the document and the rewritten clauses.

        STATUS CHECK INSTRUCTIONS:
        You may call a status check at any time using the following message format:
        "Check status of Preprocessor, Compliance Checker, and Post-Processor agents for document ID: <doc_id>"
        This returns a status dict like:
        {
          "preprocessor": "complete",
          "compliance_checker": "in_progress",
          "post_processor": "not_started"
        }

        Before initiating new tasks, always perform a status check to avoid redundant computation.

        Instructions:
        Ask the Preprocessor Agent if clause extraction and document classification are complete.
        Ask the Knowledge agent if knowledge retrieval and storage is complete.
        Ask the Clause Compliance Checker Agent if it has processed all clauses and flagged issues.
        Ask the Post Processor Agent if it has finalized the rewritten output and compliance summary.
        Use this information to update the task list, reorder tasks, or retry failed steps.

        OUTPUT:
        - An updated list of pending tasks (if any)
        - A decision on what agent should be triggered next
        - Rationale behind your planning
        """

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        document_id = state.get("document_id")
        task_ledger = state.get("task_ledger", [])
        agent_statuses = state.get("agent_statuses", {})

        # Update task ledger with returned statuses
        for entry in task_ledger:
            task = entry.get("task")
            if task == "preprocess_document":
                entry["status"] = agent_statuses.get("preprocessor", "not_started")
            elif task == "check_compliance":
                entry["status"] = agent_statuses.get("compliance_checker", "not_started")
            elif task == "generate_report":
                entry["status"] = agent_statuses.get("post_processor", "not_started")

        # Decision logic
        if agent_statuses.get("preprocessor") != "complete":
            state["next_step"] = "preprocess"
            if not any(t["task"] == "preprocess_document" for t in task_ledger):
                task_ledger.append({"task": "preprocess_document", "status": "pending", "document_id": document_id})

        elif agent_statuses.get("compliance_checker") != "complete":
            state["next_step"] = "compliance"
            if not any(t["task"] == "check_compliance" for t in task_ledger):
                task_ledger.append({"task": "check_compliance", "status": "pending", "document_id": document_id})

        else:
            contradictions = self.context_bank.get_all_contradictions(document_id)
            suggestions = self.context_bank.get_all_suggestions(document_id)

            if contradictions and not suggestions:
                state["next_step"] = "rewrite"
                if not any(t["task"] == "rewrite_clauses" for t in task_ledger):
                    task_ledger.append({
                        "task": "rewrite_clauses",
                        "status": "pending",
                        "document_id": document_id,
                        "contradiction_ids": [c.get("id") for c in contradictions]
                    })

            elif suggestions and agent_statuses.get("post_processor") != "complete":
                state["next_step"] = "postprocess"
                if not any(t["task"] == "generate_report" for t in task_ledger):
                    task_ledger.append({"task": "generate_report", "status": "pending", "document_id": document_id})

            else:
                state["next_step"] = "complete"
                state["complete"] = True

        state["task_ledger"] = task_ledger
        return state

    def _plan_tasks(self, document_id: str, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        document = self.context_bank.get_document(document_id)
        clauses = self.context_bank.clauses.get(document_id, [])

        input_text = f"""
        Document ID: {document_id}
        Document Type: {document.get('metadata', {}).get('type', 'Unknown')}
        Number of Clauses: {len(clauses)}
        Current Processing Stage: {current_state.get('current_stage', 'New Document')}

        Based on this information, create a task plan for processing this document.
        """

        response = self.llm_client.generate(
            system_prompt=self.task_planner_prompt,
            user_prompt=input_text
        )

        tasks = []
        for line in response.strip().split('\n'):
            if line.startswith('- '):
                task_desc = line[2:].strip()
                tasks.append({
                    "description": task_desc,
                    "status": "pending"
                })

        return tasks
