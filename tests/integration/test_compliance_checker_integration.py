import unittest
from unittest.mock import MagicMock, patch
import json
import os
from datetime import datetime

# Import the components we want to test
from agents.compliance_checker import ComplianceCheckerAgent
from agents.statutory_validator import StatutoryValidator, Statute, Violation, SeverityLevel
from agents.precedent_analyzer import PrecedentAnalyzer, PrecedentMatch
from agents.consistency_engine import ContractualConsistencyEngine, Inconsistency
from agents.hypergraph_analyzer import HypergraphAnalyzer, Cycle, ImpactAnalysis
from utils.ollama_client import OllamaClient
from utils.prompt_templates import PromptTemplates
from utils.response_parser import ResponseParser
from utils.confidence_scorer import ConfidenceScorer
from utils.hypergraph import LegalHypergraph
from context_bank import ContextBank


class TestComplianceCheckerIntegration(unittest.TestCase):
    """Integration tests for the Compliance Checker components and agent."""

    def setUp(self):
        """
        Set up common test fixtures.
        This includes:
        - Mock LLM client to avoid actual LLM calls
        - Context bank for shared state
        - Sample clauses and documents for testing
        """
        # Create a mock LLM client
        self.mock_llm_client = MagicMock()
        
        # Set up default response for the mock LLM
        self.mock_llm_client.generate.return_value = json.dumps({
            "issues": [
                {
                    "description": "Test issue",
                    "severity": "MEDIUM",
                    "reasoning": "This is test reasoning",
                    "references": ["Test Statute 123"],
                    "confidence": 0.85
                }
            ]
        })
        
        # Create a context bank
        self.context_bank = ContextBank()
        
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
            }
        ]
        
        # Create a sample document
        self.sample_document = {
            "id": "doc1",
            "title": "Rental Agreement",
            "metadata": {
                "jurisdiction": "California",
                "document_type": "contract"
            }
        }
        
        # Store the document and clauses in the context bank
        self.context_bank.store_document("doc1", self.sample_document)
        self.context_bank.clauses["doc1"] = self.sample_clauses
        
        # Create utility components
        self.prompt_templates = PromptTemplates()
        self.response_parser = ResponseParser()
        self.confidence_scorer = ConfidenceScorer(min_confidence_threshold=0.75)

    def test_statutory_validator_integration(self):
        """
        Test the StatutoryValidator component integration.
        
        This test verifies that:
        1. The validator correctly processes clause text
        2. It interacts properly with the LLM client
        3. It returns properly structured Violation objects
        4. The violations contain the expected data
        """
        # Configure the mock LLM response for statutory validation
        statutory_response = """
        {
            "issues": [
                {
                    "description": "Security deposit exceeds legal limit",
                    "severity": "HIGH",
                    "reasoning": "California law limits security deposits to twice the monthly rent for unfurnished units",
                    "references": ["California Civil Code § 1950.5"],
                    "confidence": 0.92
                }
            ]
        }
        """
        self.mock_llm_client.generate.return_value = statutory_response
        
        # Create a mock knowledge agent
        mock_knowledge_agent = MagicMock()
        mock_knowledge_agent.find_relevant_statutes.return_value = [
            {
                "id": "cal_civ_1950.5",
                "name": "California Civil Code",
                "section": "1950.5",
                "jurisdiction": "California",
                "text": "Security deposits for residential leases shall not exceed two months' rent for unfurnished units."
            }
        ]
        
        # Create the validator with our mocks
        validator = StatutoryValidator(
            llm_client=self.mock_llm_client,
            knowledge_agent=mock_knowledge_agent,
            min_confidence=0.75
        )
        
        # Test the validator with a sample clause
        clause_text = "The Tenant shall pay a security deposit of $5,000 upon signing this agreement."
        violations = validator.validate_clause(
            clause_text=clause_text,
            clause_id="clause1",
            jurisdiction="California"
        )
        
        # Verify the results
        self.assertIsInstance(violations, list)
        self.assertEqual(len(violations), 1)
        self.assertIsInstance(violations[0], Violation)
        self.assertEqual(violations[0].severity, SeverityLevel.HIGH)
        self.assertEqual(violations[0].statute_reference, "California Civil Code § 1950.5")
        self.assertGreaterEqual(violations[0].confidence, 0.75)
        
        # Verify the knowledge agent was called correctly
        mock_knowledge_agent.find_relevant_statutes.assert_called_once()
        args, kwargs = mock_knowledge_agent.find_relevant_statutes.call_args
        self.assertEqual(kwargs["jurisdiction"], "California")

    def test_precedent_analyzer_integration(self):
        """
        Test the PrecedentAnalyzer component integration.
        
        This test verifies that:
        1. The analyzer correctly processes clause text
        2. It interacts properly with the LLM client
        3. It returns properly structured analysis results
        4. The analysis contains the expected precedent matches and issues
        """
        # Configure the mock LLM response for precedent analysis
        precedent_response = """
        {
            "issues": [
                {
                    "description": "Clause conflicts with established case law",
                    "severity": "MEDIUM",
                    "reasoning": "In Smith v. Jones (2018), the court ruled that security deposit clauses must specify the conditions for withholding funds",
                    "references": ["Smith v. Jones, 123 Cal.App.4th 456 (2018)"],
                    "confidence": 0.88
                }
            ],
            "precedents": [
                {
                    "case_name": "Smith v. Jones",
                    "citation": "123 Cal.App.4th 456",
                    "jurisdiction": "California",
                    "year": 2018,
                    "key_holdings": ["Security deposit clauses must specify conditions for withholding funds"],
                    "relevance_score": 0.9
                }
            ]
        }
        """
        self.mock_llm_client.generate.return_value = precedent_response
        
        # Create the analyzer with our mock
        analyzer = PrecedentAnalyzer(
            llm_client=self.mock_llm_client,
            min_confidence=0.75,
            use_web_search=False
        )
        
        # Mock the _simulate_precedent_retrieval method to return test data
        analyzer._simulate_precedent_retrieval = MagicMock(return_value=[
            {
                "case_name": "Smith v. Jones",
                "citation": "123 Cal.App.4th 456",
                "jurisdiction": "California",
                "year": 2018,
                "key_holdings": ["Security deposit clauses must specify conditions for withholding funds"],
                "relevance_score": 0.9
            }
        ])
        
        # Test the analyzer with a sample clause
        clause_text = "The security deposit may be withheld at the Landlord's discretion."
        result = analyzer.analyze_precedents(
            clause_text=clause_text,
            jurisdiction="California"
        )
        
        # Verify the results
        self.assertIsInstance(result, dict)
        self.assertTrue("issues" in result)
        self.assertTrue("precedents" in result)
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(len(result["precedents"]), 1)
        self.assertEqual(result["issues"][0]["description"], "Clause conflicts with established case law")
        self.assertEqual(result["precedents"][0]["case_name"], "Smith v. Jones")
        self.assertTrue(result["has_issues"])

    def test_consistency_engine_integration(self):
        """
        Test the ContractualConsistencyEngine component integration.
        
        This test verifies that:
        1. The engine correctly processes multiple clauses
        2. It detects inconsistencies between clauses
        3. It interacts properly with the LLM client
        4. It returns properly structured analysis results
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
            use_hypergraph=True
        )
        
        # Create test clauses with an inconsistency
        test_clauses = [
            {
                "id": "clause1",
                "text": "The Tenant shall pay a security deposit of $1,000."
            },
            {
                "id": "clause2",
                "text": "The Landlord shall return the security deposit within 30 days of termination."
            },
            {
                "id": "clause4",
                "text": "Any security deposit shall be returned within 45 days of the lease ending."
            }
        ]
        
        # Test the engine
        result = engine.check_consistency(
            clauses=test_clauses,
            document_context={"jurisdiction": "California"}
        )
        
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

    def test_hypergraph_analyzer_integration(self):
        """
        Test the HypergraphAnalyzer component integration.
        
        This test verifies that:
        1. The analyzer correctly builds a hypergraph from clauses
        2. It detects cycles in the graph
        3. It identifies critical nodes
        4. It analyzes the impact of nodes
        """
        # Create the hypergraph analyzer with our mock
        analyzer = HypergraphAnalyzer(llm_client=self.mock_llm_client)
        
        # Configure the mock LLM response for cycle analysis
        cycle_response = """
        Description: Circular reference between payment terms and termination conditions
        Severity: MEDIUM
        Implications:
        - Creates ambiguity in contract execution order
        - May lead to interpretation challenges
        - Could create enforcement difficulties
        """
        self.mock_llm_client.generate.return_value = cycle_response
        
        # Create test clauses with dependencies
        test_clauses = [
            {
                "id": "clause1",
                "text": "Payment is due as specified in Section 3.",
                "heading": "Section 1: Payment Terms"
            },
            {
                "id": "clause2",
                "text": "Late fees apply as outlined in Section 1.",
                "heading": "Section 2: Late Fees"
            },
            {
                "id": "clause3",
                "text": "Payment schedule may be modified according to Section 2.",
                "heading": "Section 3: Payment Schedule"
            }
        ]
        
        # Mock the dependency analyzer's extract_references method
        analyzer.dependency_analyzer.extract_references = MagicMock(side_effect=[
            [{"type": "section", "value": "3", "full_text": "Section 3"}],  # For clause1
            [{"type": "section", "value": "1", "full_text": "Section 1"}],  # For clause2
            [{"type": "section", "value": "2", "full_text": "Section 2"}]   # For clause3
        ])
        
        # Build the graph
        graph = analyzer.build_graph(test_clauses)
        
        # Verify the graph structure
        self.assertIsInstance(graph, LegalHypergraph)
        self.assertEqual(len(graph.nodes), 3)
        self.assertGreaterEqual(len(graph.edges), 3)
        
        # Test cycle detection
        cycles = analyzer.detect_cycles(graph)
        self.assertIsInstance(cycles, list)
        self.assertGreaterEqual(len(cycles), 1)
        self.assertIsInstance(cycles[0], Cycle)
        
        # Test critical node identification
        critical_nodes = analyzer.find_critical_nodes(graph)
        self.assertIsInstance(critical_nodes, list)
        self.assertGreaterEqual(len(critical_nodes), 1)
        
        # Test impact analysis
        # Get the first node ID
        first_node_id = list(graph.nodes.keys())[0]
        impact = analyzer.analyze_impact(first_node_id, graph)
        self.assertIsInstance(impact, ImpactAnalysis)
        self.assertEqual(impact.node_id, first_node_id)
        self.assertIn("risk_level", impact.__dict__)

    def test_compliance_checker_agent_integration(self):
        """
        Test the ComplianceCheckerAgent integration with all components.
        
        This test verifies that:
        1. The agent correctly orchestrates all components
        2. It processes document state and clauses
        3. It stores results in the context bank
        4. It updates the state for the next agent
        """
        # Create mock components
        mock_statutory_validator = MagicMock()
        mock_precedent_analyzer = MagicMock()
        mock_consistency_engine = MagicMock()
        mock_hypergraph_analyzer = MagicMock()
        mock_knowledge_agent = MagicMock()
        
        # Configure mock returns
        mock_statutory_validator.validate_clause.return_value = [
            Violation(
                id="v1",
                clause_id="clause1",
                statute_reference="Test Statute 123",
                severity=SeverityLevel.MEDIUM,
                description="Test violation",
                implications=["Legal risk"],
                reasoning="Test reasoning",
                confidence=0.85
            )
        ]
        
        mock_precedent_analyzer.analyze_precedents.return_value = {
            "issues": [
                {
                    "description": "Conflicts with case law",
                    "severity": "MEDIUM",
                    "reasoning": "Test reasoning",
                    "references": ["Test Case v. Other Case"],
                    "confidence": 0.8
                }
            ],
            "precedents": [
                {
                    "case_name": "Test Case v. Other Case",
                    "citation": "123 F.3d 456",
                    "jurisdiction": "US",
                    "year": 2020,
                    "key_holdings": ["Test holding"],
                    "relevance_score": 0.9
                }
            ],
            "has_issues": True
        }
        
        mock_consistency_engine.check_consistency.return_value = {
            "inconsistencies": [
                Inconsistency(
                    id="i1",
                    source_clause_id="clause1",
                    target_clause_id="clause2",
                    description="Test inconsistency",
                    severity="MEDIUM",
                    reasoning="Test reasoning",
                    implications=["Contract interpretation issue"],
                    confidence=0.9
                )
            ],
            "has_inconsistencies": True
        }
        
        # Create the agent with our mocks
        agent = ComplianceCheckerAgent(
            use_ollama=True,
            model_name="llama2",
            context_bank=self.context_bank,
            min_confidence=0.75,
            knowledge_agent=mock_knowledge_agent
        )
        
        # Replace the components with our mocks
        agent.statutory_validator = mock_statutory_validator
        agent.precedent_analyzer = mock_precedent_analyzer
        agent.consistency_engine = mock_consistency_engine
        agent.hypergraph_analyzer = mock_hypergraph_analyzer
        
        # Create a test state
        state = {
            "document_id": "doc1",
            "step": "compliance_checker"
        }
        
        # Process the state
        result_state = agent.process(state)
        
        # Verify the component calls
        mock_statutory_validator.validate_clause.assert_called()
        mock_precedent_analyzer.analyze_precedents.assert_called()
        mock_consistency_engine.check_consistency.assert_called()
        
        # Verify the state updates
        self.assertTrue(result_state["compliance_analyzed"])
        self.assertTrue(result_state["has_contradictions"])
        self.assertEqual(result_state["next_step"], "clause_rewriter")
        
        # Verify data was stored in the context bank
        self.assertIn("document_compliance_analysis:doc1", self.context_bank.data)
        
        # Test with no contradictions
        mock_statutory_validator.validate_clause.return_value = []
        mock_precedent_analyzer.analyze_precedents.return_value = {"issues": [], "has_issues": False}
        mock_consistency_engine.check_consistency.return_value = {"inconsistencies": [], "has_inconsistencies": False}
        
        result_state = agent.process(state)
        self.assertFalse(result_state["has_contradictions"])
        self.assertEqual(result_state["next_step"], "post_processor")

    def test_check_compliance_method(self):
        """
        Test the check_compliance method of the ComplianceCheckerAgent.
        
        This test verifies that:
        1. The method correctly calls all component validators
        2. It combines results from different validators
        3. It returns a properly structured result
        4. It stores analysis in the context bank
        """
        # Create mock components
        mock_statutory_validator = MagicMock()
        mock_precedent_analyzer = MagicMock()
        mock_consistency_engine = MagicMock()
        
        # Configure mock returns
        mock_statutory_validator.validate_clause.return_value = [
            Violation(
                id="v1",
                clause_id="clause1",
                statute_reference="Test Statute 123",
                severity=SeverityLevel.MEDIUM,
                description="Test violation",
                implications=["Legal risk"],
                reasoning="Test reasoning",
                confidence=0.85
            )
        ]
        
        mock_precedent_analyzer.analyze_precedents.return_value = {
            "issues": [
                {
                    "description": "Conflicts with case law",
                    "severity": "MEDIUM",
                    "reasoning": "Test reasoning",
                    "references": ["Test Case v. Other Case"],
                    "confidence": 0.8
                }
            ],
            "has_issues": True
        }
        
        mock_consistency_engine.check_consistency.return_value = {
            "inconsistencies": [
                Inconsistency(
                    id="i1",
                    source_clause_id="clause1",
                    target_clause_id="clause2",
                    description="Test inconsistency",
                    severity="MEDIUM",
                    reasoning="Test reasoning",
                    implications=["Contract interpretation issue"],
                    confidence=0.9
                )
            ],
            "has_inconsistencies": True
        }
        
        # Create the agent with our mocks
        agent = ComplianceCheckerAgent(
            use_ollama=True,
            model_name="llama2",
            context_bank=self.context_bank,
            min_confidence=0.75
        )
        
        # Replace the components with our mocks
        agent.statutory_validator = mock_statutory_validator
        agent.precedent_analyzer = mock_precedent_analyzer
        agent.consistency_engine = mock_consistency_engine
        
        # Add datetime import to the agent module
        import datetime
        
        # Patch the datetime.now() call
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.datetime(2023, 1, 1, 12, 0, 0)
            
            # Test the check_compliance method
            result = agent.check_compliance(
                clause_text="Test clause text",
                clause_id="clause1",
                document_context={
                    "jurisdiction": "US",
                    "document_id": "doc1",
                    "clauses": [{"id": "clause2", "text": "Another clause"}]
                }
            )
        
        # Verify the component calls
        mock_statutory_validator.validate_clause.assert_called_once()
        mock_precedent_analyzer.analyze_precedents.assert_called_once()
        mock_consistency_engine.check_consistency.assert_called_once()
        
        # Verify the result structure
        self.assertIsInstance(result, dict)
        self.assertTrue(result["has_issues"])
        self.assertEqual(result["issue_count"], 3)  # 1 statutory + 1 precedent + 1 consistency
        self.assertEqual(len(result["issues"]), 3)
        
        # Verify data was stored in the context bank
        self.assertIn("compliance_analysis:clause1", self.context_bank.data)
        stored_analysis = self.context_bank.data["compliance_analysis:clause1"]
        self.assertEqual(stored_analysis["clause_id"], "clause1")
        self.assertEqual(stored_analysis["timestamp"], "2023-01-01T12:00:00")

    def test_analyze_document_structure(self):
        """
        Test the analyze_document_structure method of the ComplianceCheckerAgent.
        
        This test verifies that:
        1. The method correctly calls the hypergraph analyzer
        2. It analyzes document structure for multiple clauses
        3. It returns a properly structured result
        4. It stores analysis in the context bank
        """
        # Create a mock hypergraph analyzer
        mock_hypergraph_analyzer = MagicMock()
        
        # Configure mock returns
        mock_graph = MagicMock()
        mock_hypergraph_analyzer.build_graph.return_value = mock_graph
        
        mock_hypergraph_analyzer.detect_cycles.return_value = [
            Cycle(
                id="cycle1",
                nodes=["node1", "node2", "node3"],
                description="Test cycle",
                severity="MEDIUM",
                implications=["Logical inconsistency"]
            )
        ]
        
        mock_hypergraph_analyzer.find_critical_nodes.return_value = [
            {
                "node_id": "node1",
                "criticality_score": 8.5
            }
        ]
        
        mock_hypergraph_analyzer.analyze_relationship_clusters.return_value = [
            {
                "id": "cluster1",
                "nodes": ["node1", "node2"],
                "size": 2,
                "description": "Test cluster"
            }
        ]
        
        # Create the agent with our mock
        agent = ComplianceCheckerAgent(
            use_ollama=True,
            model_name="llama2",
            context_bank=self.context_bank,
            min_confidence=0.75
        )
        
        # Replace the hypergraph analyzer with our mock
        agent.hypergraph_analyzer = mock_hypergraph_analyzer
        
        # Test the analyze_document_structure method
        result = agent.analyze_document_structure(self.sample_clauses)
        
        # Verify the hypergraph analyzer calls
        mock_hypergraph_analyzer.build_graph.assert_called_once_with(self.sample_clauses)
        mock_hypergraph_analyzer.detect_cycles.assert_called_once_with(mock_graph)
        mock_hypergraph_analyzer.find_critical_nodes.assert_called_once_with(mock_graph)
        mock_hypergraph_analyzer.analyze_relationship_clusters.assert_called_once_with(mock_graph)
        
        # Verify the result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result["clause_count"], 3)
        self.assertEqual(result["cycle_count"], 1)
        self.assertEqual(result["critical_node_count"], 1)
        self.assertEqual(result["cluster_count"], 1)
        self.assertTrue(result["has_structural_issues"])
        
        # Verify data was stored in the context bank
        document_structure_key = f"document_structure_analysis:{result['document_id']}"
        self.assertIn(document_structure_key, self.context_bank.data)


if __name__ == '__main__':
    unittest.main() 