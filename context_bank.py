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
        self.clause_compliance_results: Dict[str, Dict[str, Any]] = {} # Store clause-level compliance results keyed by clause_id
        self.jurisdiction: Optional[str] = None # Store legal jurisdiction for the document
        self.document_analysis: Optional[Dict[str, Any]] = None # Store document-level analysis results
        self.non_compliant_clauses: List[Dict[str, Any]] = []  # Store non-compliant clauses

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
        self.clause_compliance_results = {} # Reset clause compliance results
        self.jurisdiction = None
        self.document_analysis = None # Reset document analysis

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

    def add_clause_compliance_result(self, clause_id: str, analysis: Dict[str, Any]):
        """
        Store the compliance analysis result for a specific clause.

        Args:
            clause_id: The unique identifier of the clause.
            analysis: A dictionary containing the compliance analysis details for the clause.
                      Expected to include keys like 'issues', 'issue_count', etc.
        """
        self.clause_compliance_results[clause_id] = analysis

    def get_clause_compliance_result(self, clause_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the compliance analysis result for a specific clause.

        Args:
            clause_id: The unique identifier of the clause.

        Returns:
            Optional[Dict[str, Any]]: The analysis dictionary if found, else None.
        """
        return self.clause_compliance_results.get(clause_id)

    def get_all_clause_compliance_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve all stored clause compliance analysis results.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary where keys are clause IDs
                                        and values are the analysis results.
        """
        return self.clause_compliance_results

    def add_document_analysis(self, analysis: Dict[str, Any]):
        """
        Store the document-level compliance analysis results.

        Args:
            analysis: A dictionary containing the analysis results.
                      Expected structure includes keys like 'document_id',
                      'analysis_id', 'clauses_analyzed', 'non_compliant_clauses',
                      'total_issues', 'has_issues', 'timestamp',
                      'estimated_jurisdiction', 'estimated_document_type'.
        """
        self.document_analysis = analysis

    def get_document_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the document-level compliance analysis results.

        Returns:
            Optional[Dict[str, Any]]: The analysis dictionary if stored, else None.
        """
        return self.document_analysis
    

    def add_non_compliant_clauses(self, clauses: List[Dict[str, Any]]):
        """
        Store non-compliant clauses identified during analysis.
        
        Args:
            clauses: List of non-compliant clauses with their metadata and analysis results.
        """
        self.non_compliant_clauses.extend(clauses)


    def get_all_non_compliant_clauses(self) -> List[Dict[str, Any]]:
        """
        Get all non-compliant clauses stored for the document.
            
        Returns:
            List[Dict[str, Any]]: A list of non-compliant clauses.
        """
        return self.non_compliant_clauses
