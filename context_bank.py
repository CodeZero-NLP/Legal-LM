# context_bank.py
from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime

class ContextBank:
    """
    Shared memory system accessible by all agents to store and retrieve
    document context, analysis results, and retrieved information.
    """
    
    def __init__(self):
        """Initialize an empty context bank"""
        self.documents = {}  # Store original documents and metadata
        self.entities = {}   # Store named entities
        self.clauses = {}    # Store extracted clauses
        self.laws = {}       # Store relevant laws and precedents
        self.contradictions = {}  # Store detected contradictions
        self.suggestions = {}     # Store suggested fixes
        
    def add_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> str:
        """
        Add a document to the context bank.
        
        Args:
            document_id: Unique identifier for the document
            content: Full text content of the document
            metadata: Additional information about the document
            
        Returns:
            str: The document ID
        """
        if not document_id:
            document_id = str(uuid.uuid4())
            
        self.documents[document_id] = {
            "content": content,
            "metadata": metadata,
            "added_at": datetime.now().isoformat(),
            "processed": False
        }
        return document_id
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from the context bank.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Dict or None: The document data if found
        """
        return self.documents.get(document_id)
    
    def add_entities(self, document_id: str, entities: List[Dict[str, Any]]):
        """
        Store named entities extracted from a document.
        
        Args:
            document_id: ID of the source document
            entities: List of extracted entities with their metadata
        """
        if document_id not in self.entities:
            self.entities[document_id] = []
        self.entities[document_id].extend(entities)
    
    def add_clauses(self, document_id: str, clauses: List[Dict[str, Any]]):
        """
        Store clauses extracted from a document.
        
        Args:
            document_id: ID of the source document
            clauses: List of extracted clauses with their metadata
        """
        if document_id not in self.clauses:
            self.clauses[document_id] = []
        self.clauses[document_id].extend(clauses)
    
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
    
    def add_contradiction(self, document_id: str, contradiction: Dict[str, Any]):
        """
        Store a detected contradiction.
        
        Args:
            document_id: ID of the document containing the contradiction
            contradiction: Details about the contradiction
        """
        if document_id not in self.contradictions:
            self.contradictions[document_id] = []
        self.contradictions[document_id].append(contradiction)
    
    def add_suggestion(self, document_id: str, clause_id: str, suggestion: Dict[str, Any]):
        """
        Store a suggested fix for a contradiction.
        
        Args:
            document_id: ID of the document
            clause_id: ID of the clause being fixed
            suggestion: Details about the suggested fix
        """
        key = f"{document_id}_{clause_id}"
        if key not in self.suggestions:
            self.suggestions[key] = []
        self.suggestions[key].append(suggestion)
    
    def get_all_contradictions(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all contradictions for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List: All contradictions for the document
        """
        return self.contradictions.get(document_id, [])
    
    def get_all_suggestions(self, document_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all suggestions for a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Dict: All suggestions organized by clause ID
        """
        result = {}
        for key, suggestions in self.suggestions.items():
            doc_id, clause_id = key.split('_', 1)
            if doc_id == document_id:
                result[clause_id] = suggestions
        return result
