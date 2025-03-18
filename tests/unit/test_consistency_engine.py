import unittest
from unittest.mock import Mock, patch, MagicMock
import uuid

from agents.consistency_engine import (
    ContractualConsistencyEngine, Inconsistency, Dependency, 
    DefinitionIssue, SeverityLevel
)


class TestContractualConsistencyEngine(unittest.TestCase):
    """
    Unit tests for the ContractualConsistencyEngine component.
    
    The ContractualConsistencyEngine is responsible for checking internal consistency
    within legal documents. It identifies contradictions between clauses, validates
    term definitions, and analyzes dependencies between clauses.
    """

    def setUp(self):
        """
        Set up test fixtures.
        
        Creates a mock LLM that returns a sample inconsistency between payment terms
        and initializes the consistency engine with this mock.
        """
        # Mock LLM that returns a sample inconsistency in the expected format
        self.mock_llm = Mock()
        self.mock_llm.generate = Mock(return_value="""
        [ISSUE]
        Description: Contradictory payment terms
        Severity: HIGH
        References: Clause 1, Clause 2
        Reasoning: Clause 1 specifies payment within 30 days, while Clause 2 requires payment within 15 days.
        [/ISSUE]
        """)
        
        # Initialize the engine with the mock LLM
        self.engine = ContractualConsistencyEngine(
            llm_client=self.mock_llm,
            min_confidence=0.7,
            use_hypergraph=True
        )
    
    def test_check_consistency_with_inconsistencies(self):
        """
        Test consistency check that finds inconsistencies.
        
        This test verifies that the engine correctly identifies inconsistencies
        between clauses in a document, particularly contradictory payment terms.
        
        Input: List of clauses with contradictory payment terms
        Expected output: Analysis dict with inconsistencies identified
        """
        # Arrange - Create clauses with contradictory payment terms
        clauses = [
            {"id": "clause1", "text": "Payment shall be made within 30 days of invoice receipt."},
            {"id": "clause2", "text": "All payments are due within 15 days of invoice date."}
        ]
        
        # Act - Check consistency between the clauses
        analysis = self.engine.check_consistency(clauses)
        
        # Assert - Check that inconsistencies were found with expected properties
        self.assertTrue(analysis["has_inconsistencies"])
        self.assertEqual(analysis["inconsistency_count"], 1)
        
        inconsistency = analysis["inconsistencies"][0]
        self.assertIsInstance(inconsistency, Inconsistency)
        self.assertEqual(inconsistency.severity, "HIGH")
        self.assertIn("payment terms", inconsistency.description.lower())
        self.assertIn("30 days", inconsistency.reasoning)
        self.assertIn("15 days", inconsistency.reasoning)
    
    def test_check_consistency_no_inconsistencies(self):
        """
        Test consistency check that finds no inconsistencies.
        
        This test verifies that the engine correctly returns no inconsistencies
        when clauses in a document are consistent with each other.
        
        Input: List of consistent clauses about payment terms
        Expected output: Analysis dict with no inconsistencies
        """
        # Arrange - Mock LLM to return no issues and create consistent clauses
        self.mock_llm.generate = Mock(return_value="[NO_ISSUES]")
        clauses = [
            {"id": "clause1", "text": "Payment shall be made within 30 days of invoice receipt."},
            {"id": "clause2", "text": "Late payments will incur a 1.5% monthly interest charge."}
        ]
        
        # Act - Check consistency between the consistent clauses
        analysis = self.engine.check_consistency(clauses)
        
        # Assert - Check that no inconsistencies were found
        self.assertFalse(analysis["has_inconsistencies"])
        self.assertEqual(analysis["inconsistency_count"], 0)
        self.assertEqual(len(analysis["inconsistencies"]), 0)
    
    def test_check_pair_consistency(self):
        """
        Test consistency check between a pair of clauses.
        
        This test verifies that the engine correctly identifies inconsistencies
        when checking a specific pair of clauses against each other.
        
        Input: Two clauses with contradictory payment terms
        Expected output: List containing one Inconsistency object
        """
        # Arrange - Create two clauses with contradictory payment terms
        clause1 = {"id": "clause1", "text": "Payment shall be made within 30 days of invoice receipt."}
        clause2 = {"id": "clause2", "text": "All payments are due within 15 days of invoice date."}
        
        # Act - Check consistency between the pair of clauses
        inconsistencies = self.engine.check_pair_consistency(clause1, clause2)
        
        # Assert - Check that an inconsistency was found with expected properties
        self.assertEqual(len(inconsistencies), 1)
        inconsistency = inconsistencies[0]
        self.assertEqual(inconsistency.source_clause_id, "clause1")
        self.assertEqual(inconsistency.target_clause_id, "clause2")
        self.assertEqual(inconsistency.severity, "HIGH")
    
    def test_validate_definitions(self):
        """
        Test validation of term definitions.
        
        This test verifies that the engine correctly identifies issues with term
        definitions, such as multiple conflicting definitions of the same term.
        
        Input: List of clauses with multiple definitions of "Services"
        Expected output: List containing a DefinitionIssue for "Services"
        """
        # Arrange - Create clauses with multiple definitions of "Services"
        clauses = [
            {"id": "def1", "text": "'Services' means the consulting services provided by Consultant."},
            {"id": "def2", "text": "'Services' means the software development services described in Exhibit A."},
            {"id": "clause3", "text": "Consultant shall provide the Services according to the schedule."}
        ]
        
        # Act - Validate definitions in the clauses
        definition_issues = self.engine.validate_definitions(clauses)
        
        # Assert - Check that a definition issue was found with expected properties
        self.assertGreater(len(definition_issues), 0)
        issue = definition_issues[0]
        self.assertIsInstance(issue, DefinitionIssue)
        self.assertEqual(issue.term, "Services")
        self.assertEqual(issue.issue_type, "multiple_definitions")
        self.assertIn("def1", issue.affected_clauses)
        self.assertIn("def2", issue.affected_clauses)
    
    def test_extract_defined_terms(self):
        """
        Test extraction of defined terms from clauses.
        
        This test verifies that the engine correctly extracts defined terms
        from clauses, including the term name and its definition.
        
        Input: List of clauses containing term definitions
        Expected output: Dictionary mapping terms to their definitions
        """
        # Arrange - Create clauses with term definitions
        clauses = [
            {"id": "def1", "text": "'Services' means the consulting services provided by Consultant."},
            {"id": "def2", "text": "'Client' means the party receiving the Services."}
        ]
        
        # Act - Extract defined terms from the clauses
        defined_terms = self.engine.extract_defined_terms(clauses)
        
        # Assert - Check that both terms were extracted with correct definitions
        self.assertEqual(len(defined_terms), 2)
        self.assertIn("Services", defined_terms)
        self.assertIn("Client", defined_terms)
        
        services_def = defined_terms["Services"]
        self.assertEqual(services_def["clause_id"], "def1")
        self.assertIn("consulting services", services_def["definition"])
    
    def test_analyze_dependencies(self):
        """
        Test analysis of dependencies between clauses.
        
        This test verifies that the engine correctly identifies dependencies
        between clauses, such as references to other sections of the document.
        
        Input: A clause referencing another section and all clauses in the document
        Expected output: List containing a Dependency object
        """
        # Arrange - Create a clause referencing another section
        clause = {"id": "clause1", "text": "As defined in Section 2.3, the Services include maintenance."}
        all_clauses = [
            clause,
            {"id": "clause2", "text": "Section 2.3: Maintenance Services", "heading": "Section 2.3"}
        ]
        
        # Act - Analyze dependencies for the clause
        dependencies = self.engine.analyze_dependencies(clause, all_clauses)
        
        # Assert - Check that a dependency was found with expected properties
        self.assertEqual(len(dependencies), 1)
        dependency = dependencies[0]
        self.assertIsInstance(dependency, Dependency)
        self.assertEqual(dependency.source_clause_id, "clause1")
        self.assertEqual(dependency.target_clause_id, "clause2")
        self.assertEqual(dependency.dependency_type, "Section")
    
    def test_extract_implications(self):
        """
        Test extraction of implications from inconsistency reasoning.
        
        This test verifies that the engine correctly extracts implications
        from the reasoning text about an inconsistency.
        
        Input: Reasoning text containing multiple implications
        Expected output: List of extracted implication strings
        """
        # Arrange - Create reasoning text with multiple implications
        reasoning = "This creates confusion about payment deadlines. It may lead to payment disputes."
        
        # Act - Extract implications from the reasoning
        implications = self.engine.extract_implications(reasoning)
        
        # Assert - Check that both implications were extracted
        self.assertEqual(len(implications), 2)
        self.assertIn("confusion about payment deadlines", implications[0])
        self.assertIn("payment disputes", implications[1])


if __name__ == '__main__':
    unittest.main() 