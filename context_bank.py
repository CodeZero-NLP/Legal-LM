# context_bank.py
from typing import Dict, List, Any, Optional
from datetime import datetime

class ContextBank:
    """
    Shared memory system accessible by all agents to store and retrieve
    document context, analysis results, and retrieved information for a single document.
    """
    
    def __init__(self):
        """Initialize an empty context bank for a single document"""
        self.document: Optional[Dict[str, Any]] = None # Store the single document's data
        self.entities: List[Dict[str, Any]] = []   # Store named entities
        self.clauses: List[Dict[str, Any]] = []    # Store extracted clauses
        self.laws: Dict[str, Dict[str, Any]] = {}       # Store relevant laws and precedents (assuming general)
        self.contradictions: List[Dict[str, Any]] = []  # Store detected contradictions
        self.suggestions: Dict[str, List[Dict[str, Any]]] = {}     # Store suggested fixes keyed by clause_id
        self.jurisdiction: Optional[str] = None # Store legal jurisdiction for the document
        
    def add_document(self, content: str, metadata: Dict[str, Any]):
        """
        Set the document context in the bank.
        
        Args:
            content: Full text content of the document
            metadata: Additional information about the document
        """
        self.document = {
            "content": content,
            "metadata": metadata,
            "added_at": datetime.now().isoformat(),
            "processed": False
        }
        # Reset other document-specific fields when a new document is added
        self.entities = []
        self.clauses = []
        self.contradictions = []
        self.suggestions = {}
        self.jurisdiction = None

    def get_document(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the document from the context bank.
            
        Returns:
            Dict or None: The document data if set
        """
        return self.document
    
    def add_jurisdiction(self, jurisdiction: str):
        """
        Store the legal jurisdiction for the document.

        Args:
            jurisdiction: The legal jurisdiction (e.g., "State of California", "Federal")
        """
        self.jurisdiction = jurisdiction

    def get_jurisdiction(self) -> Optional[str]:
        """
        Retrieve the legal jurisdiction for the document.

        Returns:
            str or None: The jurisdiction string if set, else None
        """
        return self.jurisdiction

    def add_entities(self, entities: List[Dict[str, Any]]):
        """
        Store named entities extracted from the document.
        
        Args:
            entities: List of extracted entities with their metadata
        """
        self.entities.extend(entities)
    
    def add_clauses(self, clauses: List[Dict[str, Any]]):
        """
        Store clauses extracted from the document.
        
        Args:
            clauses: List of extracted clauses with their metadata
        """
        self.clauses.extend(clauses)
    
    # Assuming add_law remains unchanged as laws might be general
    def add_law(self, law_id: str, content: str, source: str, metadata: Dict[str, Any]):
        """
        Add a relevant law or precedent to the context bank.
        
        Args:
            law_id: Identifier for the law (e.g., statute number)
            content: Text content of the law
            source: Source of the law (e.g., U.S. Code, case citation)
            metadata: Additional information about the law
        """
        self.laws[law_id] = {
            "content": content,
            "source": source,
            "metadata": metadata,
            "added_at": datetime.now().isoformat()
        }
    
    def add_contradiction(self, contradiction: Dict[str, Any]):
        """
        Store a detected contradiction for the document.
        
        Args:
            contradiction: Details about the contradiction
        """
        self.contradictions.append(contradiction)
    
    def add_suggestion(self, clause_id: str, suggestion: Dict[str, Any]):
        """
        Store a suggested fix for a contradiction.
        
        Args:
            clause_id: ID of the clause being fixed
            suggestion: Details about the suggested fix
        """
        if clause_id not in self.suggestions:
            self.suggestions[clause_id] = []
        self.suggestions[clause_id].append(suggestion)
    
    def get_all_contradictions(self) -> List[Dict[str, Any]]:
        """
        Get all contradictions for the document.
            
        Returns:
            List: All contradictions for the document
        """
        return self.contradictions
    
    def get_all_suggestions(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all suggestions for the document.
            
        Returns:
            Dict: All suggestions organized by clause ID
        """
        return self.suggestions
