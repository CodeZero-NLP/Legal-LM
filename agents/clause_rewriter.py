# agents/clause_rewriter.py
from typing import Dict, List, Any, Optional
import uuid
from context_bank import ContextBank
from utils.ollama_client import OllamaClient
from utils.api_client import APIClient
from utils.template_manager import TemplateManager

class ClauseRewriterAgent:
    """
    Clause Rewriter Agent that generates compliant alternatives to
    problematic clauses, preserving the original intent while ensuring
    legal compliance.
    """
    
    def __init__(self, use_ollama: bool, model_name: str, context_bank: ContextBank):
        """
        Initialize the Clause Rewriter Agent.
        
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
        
        # Initialize template manager for accessing standard clause templates
        self.template_manager = TemplateManager()
        
        # System prompt for clause rewriting
        self.rewriter_prompt = """
        You are a Legal Clause Rewriter specializing in contract drafting. Your task is to:
        1. Analyze the problematic clause and understand the contradiction or compliance issue
        2. Preserve the original business intent of the clause
        3. Rewrite the clause to resolve the legal issue while maintaining clarity
        4. Ensure the rewritten clause complies with relevant statutes and precedents
        5. Provide a brief explanation of the changes made
        
        Be precise, clear, and maintain professional legal language.
        """
        
        # System prompt for version control
        self.version_control_prompt = """
        You are a Version Control specialist for legal documents. Your task is to:
        1. Track changes between the original and rewritten clauses
        2. Identify specific modifications made to address legal issues
        3. Explain the rationale behind each change
        4. Highlight any potential new issues introduced by the changes
        
        Be thorough and precise in your analysis.
        """
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state to rewrite problematic clauses.
        
        Args:
            state: Current state of the workflow
            
        Returns:
            Dict: Updated state with rewritten clauses
        """
        document_id = state.get("document_id")
        contradictions = state.get("contradictions", [])
        
        if not document_id or not contradictions:
            state["error"] = "Missing document ID or contradictions for clause rewriting"
            state["next_step"] = "orchestrator"
            return state
        
        # Track rewritten clauses
        rewritten_clauses = []
        
        # Process each contradiction
        for contradiction in contradictions:
            # Get the relevant clause and contradiction details
            clause_id = contradiction.get("clause_id")
            clause_text = contradiction.get("clause_text")
            contradiction_type = contradiction.get("type")
            description = contradiction.get("description")
            
            # For internal contradictions, we need both clauses
            if contradiction_type == "internal":
                clause1_id = contradiction.get("clause1_id")
                clause1_text = contradiction.get("clause1_text")
                clause2_id = contradiction.get("clause2_id")
                clause2_text = contradiction.get("clause2_text")
                
                # Rewrite both clauses to resolve the internal contradiction
                rewritten_clause1 = self._rewrite_clause(
                    clause1_text, 
                    contradiction_type, 
                    description,
                    related_clause=clause2_text
                )
                
                rewritten_clause2 = self._rewrite_clause(
                    clause2_text, 
                    contradiction_type, 
                    description,
                    related_clause=clause1_text
                )
                
                # Store the suggestions
                if rewritten_clause1:
                    suggestion = {
                        "id": str(uuid.uuid4()),
                        "contradiction_id": contradiction.get("id"),
                        "original_text": clause1_text,
                        "rewritten_text": rewritten_clause1["text"],
                        "explanation": rewritten_clause1["explanation"],
                        "changes": rewritten_clause1["changes"]
                    }
                    self.context_bank.add_suggestion(document_id, clause1_id, suggestion)
                    rewritten_clauses.append(suggestion)
                
                if rewritten_clause2:
                    suggestion = {
                        "id": str(uuid.uuid4()),
                        "contradiction_id": contradiction.get("id"),
                        "original_text": clause2_text,
                        "rewritten_text": rewritten_clause2["text"],
                        "explanation": rewritten_clause2["explanation"],
                        "changes": rewritten_clause2["changes"]
                    }
                    self.context_bank.add_suggestion(document_id, clause2_id, suggestion)
                    rewritten_clauses.append(suggestion)
            
            # For statutory or precedent contradictions
            else:
                # Get relevant laws for context
                relevant_laws = []
                if contradiction_type == "statutory":
                    relevant_laws = [law for law_id, law in self.context_bank.laws.items() 
                                    if law.get("metadata", {}).get("type") == "statute"]
                elif contradiction_type == "precedent":
                    relevant_laws = [law for law_id, law in self.context_bank.laws.items() 
                                    if law.get("metadata", {}).get("type") == "precedent"]
                
                # Rewrite the clause to resolve the contradiction
                rewritten_clause = self._rewrite_clause(
                    clause_text, 
                    contradiction_type, 
                    description,
                    relevant_laws=relevant_laws
                )
                
                # Store the suggestion
                if rewritten_clause:
                    suggestion = {
                        "id": str(uuid.uuid4()),
                        "contradiction_id": contradiction.get("id"),
                        "original_text": clause_text,
                        "rewritten_text": rewritten_clause["text"],
                        "explanation": rewritten_clause["explanation"],
                        "changes": rewritten_clause["changes"]
                    }
                    self.context_bank.add_suggestion(document_id, clause_id, suggestion)
                    rewritten_clauses.append(suggestion)
        
        # Update the state with rewriting results
        state["rewritten_clauses"] = rewritten_clauses
        state["verify_rewrite"] = True  # Flag to verify the rewrites
        state["next_step"] = "compliance"  # Go back to compliance checker to verify
        
        return state
    
    def _rewrite_clause(self, clause_text: str, contradiction_type: str, 
                        description: str, related_clause: str = None, 
                        relevant_laws: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Rewrite a clause to resolve a contradiction.
        
        Args:
            clause_text: Original clause text
            contradiction_type: Type of contradiction
            description: Description of the contradiction
            related_clause: Related clause for internal contradictions
            relevant_laws: Relevant laws for context
            
        Returns:
            Dict or None: Rewritten clause with explanation and changes
        """
        # Check for similar template clauses
        template_clauses = self.template_manager.find_similar_templates(clause_text)
        
        # Prepare the input for the LLM
        input_text = f"""
        Original Clause: {clause_text}
        Contradiction Type: {contradiction_type}
        Contradiction Description: {description}
        """
        
        if related_clause:
            input_text += f"""
            Related Clause: {related_clause}
            """
        
        if relevant_laws:
            input_text += """
            Relevant Laws:
            """
            for law in relevant_laws:
                input_text += f"""
                Source: {law.get("source", "Unknown")}
                Content: {law.get("content", "")}
                """
        
        if template_clauses:
            input_text += """
            Similar Template Clauses:
            """
            for template in template_clauses:
                input_text += f"""
                Template: {template["text"]}
                """
        
        input_text += """
        Please rewrite the original clause to resolve the contradiction while preserving its business intent.
        Provide:
        1. The rewritten clause
        2. A brief explanation of the changes made
        """
        
        # Get rewritten clause from LLM
        response = self.llm_client.generate(
            system_prompt=self.rewriter_prompt,
            user_prompt=input_text
        )
        
        # Parse the response to extract the rewritten clause and explanation
        # This is a simplified implementation - in a real system, you'd want more robust parsing
        rewritten_text = ""
        explanation = ""
        
        lines = response.strip().split('\n')
        in_rewritten = False
        in_explanation = False
        
        for line in lines:
            if line.startswith("Rewritten Clause:") or line.startswith("1. Rewritten Clause:"):
                in_rewritten = True
                in_explanation = False
                continue
            elif line.startswith("Explanation:") or line.startswith("2. Explanation:"):
                in_rewritten = False
                in_explanation = True
                continue
            
            if in_rewritten:
                rewritten_text += line + "\n"
            elif in_explanation:
                explanation += line + "\n"
        
        # If we successfully extracted a rewritten clause
        if rewritten_text.strip():
            # Generate change tracking
            changes = self._track_changes(clause_text, rewritten_text.strip())
            
            return {
                "text": rewritten_text.strip(),
                "explanation": explanation.strip(),
                "changes": changes
            }
        
        return None
    
    def _track_changes(self, original_text: str, rewritten_text: str) -> str:
        """
        Track and explain changes between original and rewritten clauses.
        
        Args:
            original_text: Original clause text
            rewritten_text: Rewritten clause text
            
        Returns:
            str: Detailed explanation of changes
        """
        # Prepare the input for the LLM
        input_text = f"""
        Original Text: {original_text}
        
        Rewritten Text: {rewritten_text}
        
        Please identify and explain the specific changes made between these two versions.
        Focus on substantive legal modifications rather than stylistic changes.
        """
        
        # Get change tracking from LLM
        response = self.llm_client.generate(
            system_prompt=self.version_control_prompt,
            user_prompt=input_text
        )
        
        return response
