# agents/postprocessor.py
from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime
from context_bank import ContextBank
from utils.ollama_client import OllamaClient
from utils.api_client import APIClient

class PostprocessorAgent:
    """
    Post-processor Agent that creates summaries, lists changes,
    highlights risks averted, and provides references for the
    legal document review process.
    """
    
    def __init__(self, use_ollama: bool, model_name: str, context_bank: ContextBank):
        """
        Initialize the Post-processor Agent.
        
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
        
        # System prompt for explanation generation
        self.explanation_generator_prompt = """
        You are an Explanation Generator specializing in legal document summaries. Your task is to:
        1. Create a comprehensive summary of the legal document
        2. List all changes made to resolve contradictions
        3. Highlight the risks that were averted through these changes
        4. Provide references to relevant laws and precedents
        
        Be clear, concise, and focus on the most important aspects of the document and changes.
        """
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state to generate the final report.
        
        Args:
            state: Current state of the workflow
            
        Returns:
            Dict: Updated state with final report
        """
        document_id = state.get("document_id")
        document = self.context_bank.get_document(document_id)
        
        if not document:
            state["error"] = "Missing document for post-processing"
            state["next_step"] = "orchestrator"
            return state
        
        # Get all clauses, contradictions, and suggestions from the context bank
        clauses = self.context_bank.clauses.get(document_id, [])
        contradictions = self.context_bank.get_all_contradictions(document_id)
        suggestions = self.context_bank.get_all_suggestions(document_id)
        
        # Generate the final report
        report = self._generate_report(document, clauses, contradictions, suggestions)
        
        # Update the state with the report
        state["report"] = report
        state["report_generated"] = True
        state["complete"] = True
        state["next_step"] = "complete"
        
        return state
    
    def _generate_report(self, document: Dict[str, Any], clauses: List[Dict[str, Any]], 
                        contradictions: List[Dict[str, Any]], 
                        suggestions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate a comprehensive report of the document analysis.
        
        Args:
            document: Document data
            clauses: Extracted clauses
            contradictions: Detected contradictions
            suggestions: Suggested fixes
            
        Returns:
            Dict: Comprehensive report
        """
        # Prepare document summary
        document_summary = self._generate_document_summary(document, clauses)
        
        # Prepare changes summary
        changes_summary = self._generate_changes_summary(contradictions, suggestions)
        
        # Prepare risks averted summary
        risks_summary = self._generate_risks_summary(contradictions, suggestions)
        
        # Prepare references
        references = self._generate_references()
        
        # Combine into final report
        report = {
            "id": str(uuid.uuid4()),
            "document_id": document.get("id", ""),
            "document_title": document.get("metadata", {}).get("title", "Untitled Document"),
            "generated_at": datetime.now().isoformat(),
            "document_summary": document_summary,
            "changes_summary": changes_summary,
            "risks_summary": risks_summary,
            "references": references
        }
        
        return report
    
    def _generate_document_summary(self, document: Dict[str, Any], clauses: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of the document.
        
        Args:
            document: Document data
            clauses: Extracted clauses
            
        Returns:
            str: Document summary
        """
        # Prepare the input for the LLM
        document_content = document.get("content", "")
        document_type = document.get("metadata", {}).get("document_type", "Unknown")
        
        # For very long documents, we'll summarize by sections
        if len(document_content) > 10000:
            # Extract a representative subset of the document
            document_sample = document_content[:5000] + "...\n[Content truncated]...\n" + document_content[-5000:]
        else:
            document_sample = document_content
        
        input_text = f"""
        Document Type: {document_type}
        Document Content: {document_sample}
        
        Number of Clauses: {len(clauses)}
        
        Please generate a comprehensive summary of this legal document, including:
        1. The main purpose of the document
        2. Key parties involved
        3. Major obligations and rights
        4. Important terms and conditions
        5. Any notable or unusual provisions
        """
        
        # Get summary from LLM
        response = self.llm_client.generate(
            system_prompt=self.explanation_generator_prompt,
            user_prompt=input_text
        )
        
        return response
    
    def _generate_changes_summary(self, contradictions: List[Dict[str, Any]], 
                                suggestions: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Generate a summary of changes made to resolve contradictions.
        
        Args:
            contradictions: Detected contradictions
            suggestions: Suggested fixes
            
        Returns:
            str: Changes summary
        """
        # If no contradictions or suggestions, return appropriate message
        if not contradictions or not suggestions:
            return "No changes were necessary as no contradictions were detected in the document."
        
        # Prepare the input for the LLM
        input_text = "Changes Made to Resolve Contradictions:\n\n"
        
        for contradiction in contradictions:
            contradiction_id = contradiction.get("id")
            contradiction_type = contradiction.get("type")
            contradiction_desc = contradiction.get("description")
            
            input_text += f"Contradiction ({contradiction_type}):\n{contradiction_desc}\n\n"
            
            # Find suggestions for this contradiction
            related_suggestions = []
            for clause_id, clause_suggestions in suggestions.items():
                for suggestion in clause_suggestions:
                    if suggestion.get("contradiction_id") == contradiction_id:
                        related_suggestions.append(suggestion)
            
            if related_suggestions:
                input_text += "Changes Made:\n"
                for suggestion in related_suggestions:
                    input_text += f"- Original: {suggestion.get('original_text')[:100]}...\n"
                    input_text += f"- Rewritten: {suggestion.get('rewritten_text')[:100]}...\n"
                    input_text += f"- Explanation: {suggestion.get('explanation')}\n\n"
            else:
                input_text += "No changes were implemented for this contradiction.\n\n"
        
        input_text += """
        Please summarize the changes made to resolve the contradictions in this document.
        Focus on the most significant changes and their impact on the document's legal effect.
        """
        
        # Get changes summary from LLM
        response = self.llm_client.generate(
            system_prompt=self.explanation_generator_prompt,
            user_prompt=input_text
        )
        
        return response
    
    def _generate_risks_summary(self, contradictions: List[Dict[str, Any]], 
                               suggestions: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Generate a summary of risks averted through changes.
        
        Args:
            contradictions: Detected contradictions
            suggestions: Suggested fixes
            
        Returns:
            str: Risks summary
        """
        # If no contradictions or suggestions, return appropriate message
        if not contradictions or not suggestions:
            return "No significant legal risks were identified in the document."
        
        # Prepare the input for the LLM
        input_text = "Risks Averted Through Changes:\n\n"
        
        for contradiction in contradictions:
            contradiction_id = contradiction.get("id")
            contradiction_type = contradiction.get("type")
            implications = contradiction.get("implications", "")
            severity = contradiction.get("severity", "medium")
            
            input_text += f"Risk ({contradiction_type}, Severity: {severity}):\n{implications}\n\n"
            
            # Find suggestions for this contradiction
            related_suggestions = []
            for clause_id, clause_suggestions in suggestions.items():
                for suggestion in clause_suggestions:
                    if suggestion.get("contradiction_id") == contradiction_id:
                        related_suggestions.append(suggestion)
            
            if related_suggestions:
                input_text += "How Risk Was Addressed:\n"
                for suggestion in related_suggestions:
                    input_text += f"- {suggestion.get('explanation')}\n\n"
            else:
                input_text += "This risk was identified but not addressed.\n\n"
        
        input_text += """
        Please summarize the legal risks that were averted through the changes made to this document.
        Explain the potential consequences that could have occurred if these issues had not been addressed.
        Focus on practical business and legal implications.
        """
        
        # Get risks summary from LLM
        response = self.llm_client.generate(
            system_prompt=self.explanation_generator_prompt,
            user_prompt=input_text
        )
        
        return response
    
    def _generate_references(self) -> List[Dict[str, str]]:
        """
        Generate references to relevant laws and precedents.
        
        Returns:
            List: References to laws and precedents
        """
        references = []
        
        # Add all laws from the context bank as references
        for law_id, law in self.context_bank.laws.items():
            references.append({
                "id": law_id,
                "type": law.get("metadata", {}).get("type", "unknown"),
                "title": law.get("source", "Unknown Source"),
                "content": law.get("content", "")[:200] + "...",  # Truncate for brevity
                "url": law.get("metadata", {}).get("url", "")
            })
        
        return references
