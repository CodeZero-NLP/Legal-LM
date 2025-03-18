import unittest
from unittest.mock import MagicMock, patch
import json
import uuid

from agents.consistency_engine import ContractualConsistencyEngine, Inconsistency, Dependency, DefinitionIssue, SeverityLevel
from utils.ollama_client import OllamaClient
from utils.prompt_templates import PromptTemplates
from utils.response_parser import ResponseParser
from utils.confidence_scorer import ConfidenceScorer
from utils.dependency_analyzer import DependencyAnalyzer
from utils.hypergraph import LegalHypergraph


class TestConsistencyEngineIntegration(unittest.TestCase):
    """
    Integration tests for the ContractualConsistencyEngine component.
    
    These tests verify that the ConsistencyEngine correctly:
    1. Detects inconsistencies between clauses
    2. Identifies definition issues
    3. Analyzes dependencies between clauses
    4. Integrates with the hypergraph analyzer for complex relationships
    """

    def setUp(self):
        """
        Set up common test fixtures.
        This includes:
        - Mock LLM client to avoid actual LLM calls
        - Mock dependency analyzer
        - Sample clauses with various inconsistencies
        """
        # Create a mock LLM client
        self.mock_llm_client = MagicMock()
        
        # Set up default response for the mock LLM
        self.mock_llm_client.generate.return_value = json.dumps({
            "issues": [
                {
                    "description": "Test inconsistency",
                    "severity": "MEDIUM",
                    "reasoning": "This is test reasoning",
                    "references": ["Clause A", "Clause B"],
                    "confidence": 0.85
                }
            ]
        })
        
        # Create a mock dependency analyzer
        self.mock_dependency_analyzer = MagicMock()
        
        # Create sample clauses for testing
        self.sample_clauses = [
            {
                "id": "clause1",
                "text": "The Tenant shall pay a security deposit of $1,000 upon signing this agreement."
            },
            {
                "id": "clause2",
                "text": "The Landlord shall return the security deposit within 30 days of the termination of this agreement."
            },
            {
                "id": "clause3",
                "text": "The security deposit shall be used to cover any damages beyond normal wear and tear."
            },
            {
                "id": "clause4",
                "text": "Any security deposit shall be returned within 45 days of the lease ending."
            },
            {
                "id": "clause5",
                "text": "\"Security Deposit\" means funds held by the Landlord as security for the Tenant's performance."
            }
        ]
        
        # Create clauses with definition issues
        self.definition_clauses = [
            {
                "id": "def1",
                "text": "\"Term\" means the period from the Commencement Date to the Expiration Date."
            },
            {
                "id": "def2",
                "text": "The \"Term\" shall be defined as the initial lease period of 12 months."
            },
            {
                "id": "clause1",
                "text": "During the Term, Tenant shall pay rent on the first day of each month."
            }
        ]
        
        # Create utility components
        self.prompt_templates = PromptTemplates()
        self.response_parser = ResponseParser()
        self.confidence_scorer = ConfidenceScorer(min_confidence_threshold=0.75)

    def test_check_consistency_with_inconsistent_clauses(self):
        """
        Test the check_consistency method with clauses containing inconsistencies.
        
        This test verifies that:
        1. The engine correctly identifies inconsistencies between clauses
        2. It formats the prompt with the clause pairs
        3. It processes the LLM response correctly
        4. It returns properly structured inconsistency objects
        """
        # Configure the mock LLM response for consistency checking
        consistency_response = """
        {
            "issues": [
                {
                    "description": "Inconsistent timeframes for security deposit return",
                    "severity": "MEDIUM",
                    "reasoning": "Clause 2 states 30 days for return, but Clause 4 states 45 days",
                    "references": ["Clause 2", "Clause 4"],
                    "confidence": 0.95
                }
            ]
        }
        """
        self.mock_llm_client.generate.return_value = consistency_response
        
        # Create the consistency engine with our mock
        engine = ContractualConsistencyEngine(
            llm_client=self.mock_llm_client,
            min_confidence=0.75,
            use_hypergraph=False  # Disable hypergraph for this test
        )
        
        # Replace the dependency analyzer with our mock
        engine.dependency_analyzer = self.mock_dependency_analyzer
        
        # Configure the mock dependency analyzer
        mock_graph = MagicMock()
        self.mock_dependency_analyzer.build_dependency_graph.return_value = mock_graph
        mock_graph.detect_cycles.return_value = []
        self.mock_dependency_analyzer.find_long_range_dependencies.return_value = []
        
        # Test the engine with clauses containing an inconsistency
        result = engine.check_consistency(
            clauses=[self.sample_clauses[1], self.sample_clauses[3]],  # Clauses 2 and 4 with different timeframes
            document_context={"jurisdiction": "California"}
        )
        
        # Verify the dependency analyzer was called
        self.mock_dependency_analyzer.build_dependency_graph.assert_called_once()
        
        # Verify the LLM was called with the right prompt
        self.mock_llm_client.generate.assert_called()
        args, kwargs = self.mock_llm_client.generate.call_args
        self.assertIn("security deposit", kwargs["user_prompt"])
        self.assertIn("30 days", kwargs["user_prompt"])
        self.assertIn("45 days", kwargs["user_prompt"])
        
        # Verify the results
        self.assertIsInstance(result, dict)
        self.assertTrue("inconsistencies" in result)
        self.assertTrue("has_inconsistencies" in result)
        self.assertTrue(result["has_inconsistencies"])
        self.assertEqual(len(result["inconsistencies"]), 1)
        
        # Check the inconsistency details
        inconsistency = result["inconsistencies"][0]
        self.assertIsInstance(inconsistency, Inconsistency)
        self.assertEqual(inconsistency.description, "Inconsistent timeframes for security deposit return")
        self.assertEqual(inconsistency.severity, "MEDIUM")
        self.assertGreaterEqual(inconsistency.confidence, 0.75)

    def test_check_consistency_with_hypergraph(self):
        """
        Test the check_consistency method with hypergraph analysis enabled.
        
        This test verifies that:
        1. The engine correctly uses the hypergraph for cycle detection
        2. It identifies cycle-based inconsistencies
        3. It processes both pairwise and cycle-based inconsistencies
        4. It returns a comprehensive analysis
        """
        # Configure the mock LLM responses
        # First for pairwise comparison
        pairwise_response = """
        {
            "issues": [
                {
                    "description": "Minor terminology inconsistency",
                    "severity": "LOW",
                    "reasoning": "Clause 2 uses 'termination' while Clause 4 uses 'lease ending'",
                    "references": ["Clause 2", "Clause 4"],
                    "confidence": 0.8
                }
            ]
        }
        """
        # Then for cycle analysis
        cycle_response = """
        Description: Circular reference creating logical inconsistency
        Severity: HIGH
        Implications:
        - Creates ambiguity in contract execution
        - May lead to interpretation challenges
        - Could create enforcement difficulties
        """
        
        self.mock_llm_client.generate.side_effect = [pairwise_response, cycle_response]
        
        # Create the consistency engine with hypergraph enabled
        engine = ContractualConsistencyEngine(
            llm_client=self.mock_llm_client,
            min_confidence=0.75,
            use_hypergraph=True
        )
        
        # Replace the dependency analyzer with our mock
        engine.dependency_analyzer = self.mock_dependency_analyzer
        
        # Configure the mock dependency analyzer to return a cycle
        mock_graph = MagicMock()
        self.mock_dependency_analyzer.build_dependency_graph.return_value = mock_graph
        mock_graph.detect_cycles.return_value = [["0", "1", "2"]]  # A cycle involving 3 nodes
        self.mock_dependency_analyzer.find_long_range_dependencies.return_value = [
            {"source": "0", "target": "2", "path_length": 2}
        ]
        
        # Test the engine
        result = engine.check_consistency(
            clauses=self.sample_clauses[:3],  # First 3 clauses
            document_context={"jurisdiction": "California"}
        )
        
        # Verify the dependency analyzer was called
        self.mock_dependency_analyzer.build_dependency_graph.assert_called_once()
        mock_graph.detect_cycles.assert_called_once()
        
        # Verify the results
        self.assertIsInstance(result, dict)
        self.assertTrue("inconsistencies" in result)
        self.assertTrue("has_inconsistencies" in result)
        self.assertTrue(result["has_inconsistencies"])
        self.assertGreaterEqual(len(result["inconsistencies"]), 1)
        self.assertTrue("cycles" in result)
        self.assertEqual(len(result["cycles"]), 1)
        
        # Check for cycle-based inconsistency
        cycle_inconsistency = None
        for inconsistency in result["inconsistencies"]:
            if "circular" in inconsistency.description.lower():
                cycle_inconsistency = inconsistency
                break
                
        self.assertIsNotNone(cycle_inconsistency)
        self.assertEqual(cycle_inconsistency.severity, "HIGH")

    def test_validate_definitions(self):
        """
        Test the _validate_definitions method.
        
        This test verifies that:
        1. The engine correctly identifies defined terms
        2. It detects multiple inconsistent definitions
        3. It analyzes term usage across clauses
        4. It returns properly structured definition issues
        """
        # Configure the mock LLM responses
        # First for definition extraction
        definition_extraction_response = """
        [TERM]
        Term: Term
        Definition: the period from the Commencement Date to the Expiration Date
        [/TERM]
        """
        
        # Then for definition consistency check
        consistency_check_response = "INCONSISTENT"
        
        self.mock_llm_client.generate.side_effect = [
            definition_extraction_response,  # For first clause
            definition_extraction_response,  # For second clause
            "NO_DEFINED_TERMS",              # For third clause
            consistency_check_response       # For consistency check
        ]
        
        # Create the consistency engine
        engine = ContractualConsistencyEngine(
            llm_client=self.mock_llm_client,
            min_confidence=0.75
        )
        
        # Test the _validate_definitions method directly
        definition_issues = engine._validate_definitions(self.definition_clauses)
        
        # Verify the LLM was called for each clause and for consistency check
        self.assertEqual(self.mock_llm_client.generate.call_count, 4)
        
        # Verify the results
        self.assertIsInstance(definition_issues, list)
        self.assertEqual(len(definition_issues), 1)
        self.assertIsInstance(definition_issues[0], DefinitionIssue)
        self.assertEqual(definition_issues[0].term, "Term")
        self.assertEqual(definition_issues[0].issue_type, "multiple_definitions")
        self.assertEqual(len(definition_issues[0].affected_clauses), 2)

    def test_analyze_dependencies(self):
        """
        Test the analyze_dependencies method.
        
        This test verifies that:
        1. The engine correctly extracts references from clauses
        2. It maps references to target clauses
        3. It identifies different types of dependencies
        4. It returns properly structured Dependency objects
        """
        # Create the consistency engine
        engine = ContractualConsistencyEngine(
            llm_client=self.mock_llm_client,
            min_confidence=0.75
        )
        
        # Mock the dependency analyzer's extract_references method
        engine.dependency_analyzer.extract_references = MagicMock(return_value=[
            {
                "type": "section",
                "value": "3",
                "full_text": "Section 3"
            }
        ])
        
        # Mock the _reference_matches_clause method
        engine._reference_matches_clause = MagicMock(return_value=True)
        
        # Create test clauses with references
        test_clauses = [
            {
                "id": "clause1",
                "text": "Payment is due as specified in Section 3."
            },
            {
                "id": "clause3",
                "text": "Section 3: Payment Schedule. Payments shall be made monthly."
            }
        ]
        
        # Test the analyze_dependencies method
        dependencies = engine.analyze_dependencies(
            clause=test_clauses[0],
            all_clauses=test_clauses
        )
        
        # Verify the dependency analyzer was called
        engine.dependency_analyzer.extract_references.assert_called_once_with(test_clauses[0]["text"])
        
        # Verify the _reference_matches_clause method was called
        engine._reference_matches_clause.assert_called_once()
        
        # Verify the results
        self.assertIsInstance(dependencies, list)
        self.assertEqual(len(dependencies), 1)
        self.assertIsInstance(dependencies[0], Dependency)
        self.assertEqual(dependencies[0].source_clause_id, "clause1")
        self.assertEqual(dependencies[0].target_clause_id, "clause3")
        self.assertEqual(dependencies[0].dependency_type, "section")


if __name__ == '__main__':
    unittest.main() 