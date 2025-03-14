# agents/knowledge.py
from typing import Dict, List, Any, Optional
import requests
import json
from context_bank import ContextBank
from utils.ollama_client import OllamaClient
from utils.api_client import APIClient
from utils.web_searcher import WebSearcher
from utils.statute_finder import StatuteFinder

class KnowledgeAgent:
    """
    Knowledge Agent that retrieves relevant legal information, statutes,
    precedents, and gold-standard documents to provide context for analysis.
    """
    
    def __init__(self, use_ollama: bool, model_name: str, context_bank: ContextBank):
        """
        Initialize the Knowledge Agent with its sub-components.
        
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
        
        # Initialize sub-components
        self.web_searcher = WebSearcher()
        self.statute_finder = StatuteFinder()
        
        # System prompt for knowledge retrieval
        self.knowledge_prompt = """
        You are a Legal Knowledge Retrieval specialist. Your task is to:
        1. Identify key legal concepts, statutes, and precedents relevant to the given legal text
        2. Formulate precise search queries to find applicable laws and regulations
        3. Extract the most relevant information from search results
        4. Summarize the legal context that applies to the given text
        
        Be thorough, accurate, and focus on the most relevant legal information.
        """
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state to retrieve relevant legal knowledge.
        
        Args:
            state: Current state of the workflow
            
        Returns:
            Dict: Updated state with retrieved knowledge
        """
        document_id = state.get("document_id")
        document = self.context_bank.get_document(document_id)
        clauses = self.context_bank.clauses.get(document_id, [])
        
        if not document or not clauses:
            state["error"] = "Missing document or clauses for knowledge retrieval"
            state["next_step"] = "orchestrator"
            return state
        
        # Track retrieved knowledge
        retrieved_laws = []
        
        # Process each clause to find relevant legal knowledge
        for clause in clauses:
            # Skip if already processed
            if "knowledge_retrieved" in clause and clause["knowledge_retrieved"]:
                continue
                
            # Identify relevant legal concepts for this clause
            legal_concepts = self._identify_legal_concepts(clause["text"], clause["type"])
            
            # For each concept, retrieve relevant statutes and precedents
            for concept in legal_concepts:
                # Search for statutes
                statutes = self._find_statutes(concept)
                for statute in statutes:
                    law_id = statute.get("id")
                    if law_id:
                        self.context_bank.add_law(
                            law_id=law_id,
                            content=statute.get("text", ""),
                            source=statute.get("source", ""),
                            metadata={
                                "type": "statute",
                                "jurisdiction": statute.get("jurisdiction", ""),
                                "effective_date": statute.get("effective_date", ""),
                                "related_clause_id": clause["id"]
                            }
                        )
                        retrieved_laws.append(law_id)
                
                # Search for precedents
                precedents = self._find_precedents(concept)
                for precedent in precedents:
                    law_id = precedent.get("id")
                    if law_id:
                        self.context_bank.add_law(
                            law_id=law_id,
                            content=precedent.get("text", ""),
                            source=precedent.get("source", ""),
                            metadata={
                                "type": "precedent",
                                "court": precedent.get("court", ""),
                                "date": precedent.get("date", ""),
                                "related_clause_id": clause["id"]
                            }
                        )
                        retrieved_laws.append(law_id)
            
            # Mark this clause as processed
            clause["knowledge_retrieved"] = True
        
        # Update the state with knowledge retrieval results
        state["knowledge_retrieved"] = True
        state["retrieved_laws"] = retrieved_laws
        state["next_step"] = "compliance"  # Next, check for compliance issues
        
        return state
    
    def _identify_legal_concepts(self, clause_text: str, clause_type: str) -> List[Dict[str, Any]]:
        """
        Identify key legal concepts in a clause that need further research.
        
        Args:
            clause_text: Text of the clause
            clause_type: Type/classification of the clause
            
        Returns:
            List: Legal concepts identified in the clause
        """
        # Prepare the input for the LLM
        input_text = f"""
        Clause Text: {clause_text}
        Clause Type: {clause_type}
        
        Identify the key legal concepts in this clause that require retrieval of relevant statutes and precedents.
        For each concept, provide:
        1. The concept name
        2. Relevant jurisdiction (if apparent)
        3. Specific legal areas involved
        4. Keywords for searching statutes and precedents
        """
        
        # Get concepts from LLM
        response = self.llm_client.generate(
            system_prompt=self.knowledge_prompt,
            user_prompt=input_text
        )
        
        # Parse the response to extract concepts
        # This is a simplified implementation - in a real system, you'd want more robust parsing
        concepts = []
        current_concept = {}
        
        for line in response.strip().split('\n'):
            if line.startswith('Concept:') or line.startswith('- Concept:'):
                if current_concept and 'name' in current_concept:
                    concepts.append(current_concept)
                current_concept = {"name": line.split(':', 1)[1].strip()}
            elif line.startswith('Jurisdiction:') or line.startswith('- Jurisdiction:'):
                current_concept["jurisdiction"] = line.split(':', 1)[1].strip()
            elif line.startswith('Legal Areas:') or line.startswith('- Legal Areas:'):
                current_concept["legal_areas"] = line.split(':', 1)[1].strip()
            elif line.startswith('Keywords:') or line.startswith('- Keywords:'):
                current_concept["keywords"] = line.split(':', 1)[1].strip()
        
        # Add the last concept if it exists
        if current_concept and 'name' in current_concept:
            concepts.append(current_concept)
        
        return concepts
    
    def _find_statutes(self, legal_concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find relevant statutes for a legal concept.
        
        Args:
            legal_concept: Legal concept to search for
            
        Returns:
            List: Relevant statutes
        """
        # Prepare search query
        jurisdiction = legal_concept.get("jurisdiction", "federal")
        keywords = legal_concept.get("keywords", legal_concept.get("name", ""))
        
        # Use the statute finder to search for relevant statutes
        statutes = self.statute_finder.search(
            keywords=keywords,
            jurisdiction=jurisdiction
        )
        
        return statutes
    
    def _find_precedents(self, legal_concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find relevant precedents for a legal concept.
        
        Args:
            legal_concept: Legal concept to search for
            
        Returns:
            List: Relevant precedents
        """
        # Prepare search query
        keywords = legal_concept.get("keywords", legal_concept.get("name", ""))
        
        # Use web searcher to find precedents
        search_query = f"{keywords} legal precedent case law"
        search_results = self.web_searcher.search(search_query)
        
        # Process search results to extract precedents
        precedents = []
        for result in search_results:
            # Check if the result appears to be a legal case
            if self._is_legal_precedent(result["title"], result["snippet"]):
                precedents.append({
                    "id": f"precedent_{len(precedents)}",
                    "title": result["title"],
                    "text": result["snippet"],
                    "source": result["url"],
                    "court": self._extract_court(result["title"], result["snippet"]),
                    "date": self._extract_date(result["title"], result["snippet"])
                })
        
        return precedents
    
    def _is_legal_precedent(self, title: str, snippet: str) -> bool:
        """
        Determine if a search result is likely a legal precedent.
        
        Args:
            title: Title of the search result
            snippet: Snippet of the search result
            
        Returns:
            bool: True if the result appears to be a legal precedent
        """
        # Check for common patterns in legal case citations
        case_patterns = [
            "v.", "vs.", "versus",
            "court", "supreme court", "circuit", "district court",
            "case", "opinion", "decision", "ruling"
        ]
        
        # Check if any pattern is in the title or snippet
        for pattern in case_patterns:
            if pattern.lower() in title.lower() or pattern.lower() in snippet.lower():
                return True
        
        return False
    
    def _extract_court(self, title: str, snippet: str) -> str:
        """
        Extract the court name from a precedent.
        
        Args:
            title: Title of the precedent
            snippet: Snippet of the precedent
            
        Returns:
            str: Extracted court name or empty string
        """
        # Common court names to look for
        court_patterns = [
            "Supreme Court", "Circuit Court", "District Court",
            "Court of Appeals", "Federal Court", "State Court"
        ]
        
        # Check for court names in title and snippet
        combined_text = f"{title} {snippet}"
        for court in court_patterns:
            if court in combined_text:
                return court
        
        return ""
    
    def _extract_date(self, title: str, snippet: str) -> str:
        """
        Extract the date from a precedent.
        
        Args:
            title: Title of the precedent
            snippet: Snippet of the precedent
            
        Returns:
            str: Extracted date or empty string
        """
        # This is a simplified implementation
        # In a real system, you'd want to use regex or a date extraction library
        import re
        
        # Look for year patterns (1900-2099)
        combined_text = f"{title} {snippet}"
        year_match = re.search(r'\b(19|20)\d{2}\b', combined_text)
        
        if year_match:
            return year_match.group(0)
        
        return ""
