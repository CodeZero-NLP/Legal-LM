import unittest
from unittest.mock import MagicMock, patch
import json
import uuid
import re

from agents.hypergraph_analyzer import HypergraphAnalyzer, Cycle, ImpactAnalysis, RelationshipType
from utils.ollama_client import OllamaClient
from utils.hypergraph import LegalHypergraph
from utils.dependency_analyzer import DependencyAnalyzer


class TestHypergraphAnalyzerIntegration(unittest.TestCase):
    """
    Integration tests for the HypergraphAnalyzer component.
    
    These tests verify that the HypergraphAnalyzer correctly:
    1. Builds hypergraphs from legal clauses
    2. Detects cycles and analyzes their implications
    3. Identifies critical nodes in the graph
    4. Analyzes the impact of changes to specific nodes
    5. Identifies clusters of related clauses
    """

    def setUp(self):
        """
        Set up common test fixtures.
        This includes:
        - Mock LLM client to avoid actual LLM calls
        - Mock dependency analyzer
        - Sample clauses with various relationships
        """
        # Create a mock LLM client
        self.mock_llm_client = MagicMock()
        
        # Set up default response for the mock LLM
        self.mock_llm_client.generate.return_value = """
        Description: Test cycle description
        Severity: MEDIUM
        Implications:
        - Implication 1
        - Implication 2
        """
        
        # Create a mock dependency analyzer
        self.mock_dependency_analyzer = MagicMock()
        
        # Create sample clauses for testing
        self.sample_clauses = [
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
        
        # Create clauses with definitions
        self.definition_clauses = [
            {
                "id": "def1",
                "text": "\"Term\" means the period from the Commencement Date to the Expiration Date.",
                "heading": "Section 1: Definitions"
            },
            {
                "id": "clause2",
                "text": "During the Term, Tenant shall pay rent on the first day of each month.",
                "heading": "Section 2: Rent Payment"
            }
        ]
        
        # Create clauses with conditional language
        self.conditional_clauses = [
            {
                "id": "clause1",
                "text": "If Tenant fails to pay rent, Landlord may charge late fees.",
                "heading": "Section 1: Late Fees"
            },
            {
                "id": "clause2",
                "text": "Provided that Tenant has given notice, repairs shall be made by Landlord.",
                "heading": "Section 2: Repairs"
            },
            {
                "id": "clause3",
                "text": "Notwithstanding Section 1, no late fees shall apply during the first month.",
                "heading": "Section 3: Grace Period"
            }
        ]

    def test_build_graph(self):
        """
        Test the build_graph method.
        
        This test verifies that:
        1. The analyzer correctly builds a hypergraph from clauses
        2. It identifies references between clauses
        3. It creates appropriate edges with relationship types
        4. The resulting graph has the expected structure
        """
        # Create the hypergraph analyzer
        analyzer = HypergraphAnalyzer(llm_client=self.mock_llm_client)
        
        # Replace the dependency analyzer with our mock
        analyzer.dependency_analyzer = self.mock_dependency_analyzer
        
        # Configure the mock dependency analyzer to return references
        analyzer.dependency_analyzer.extract_references.side_effect = [
            [{"type": "section", "value": "3", "full_text": "Section 3"}],  # For clause1
            [{"type": "section", "value": "1", "full_text": "Section 1"}],  # For clause2
            [{"type": "section", "value": "2", "full_text": "Section 2"}]   # For clause3
        ]
        
        # Mock the _reference_matches_clause method to control matching
        original_method = analyzer._reference_matches_clause
        analyzer._reference_matches_clause = MagicMock(side_effect=lambda ref, clause: 
            (ref["type"] == "section" and 
             ref["value"] == clause["heading"].split(":")[0].split(" ")[1])
        )
        
        # Build the graph
        graph = analyzer.build_graph(self.sample_clauses)
        
        # Verify the dependency analyzer was called for each clause
        self.assertEqual(analyzer.dependency_analyzer.extract_references.call_count, 3)
        
        # Verify the graph structure
        self.assertIsInstance(graph, LegalHypergraph)
        self.assertEqual(len(graph.nodes), 3)
        
        # Count edges (should have 3 reference edges)
        edge_count = 0
        reference_edges = 0
        for edge_id, edge in graph.edges.items():
            edge_count += 1
            if edge.edge_type == RelationshipType.REFERENCE.value:
                reference_edges += 1
        
        self.assertEqual(edge_count, 3)
        self.assertEqual(reference_edges, 3)
        
        # Restore the original method
        analyzer._reference_matches_clause = original_method

    def test_detect_cycles(self):
        """
        Test the detect_cycles method.
        
        This test verifies that:
        1. The analyzer correctly detects cycles in the graph
        2. It analyzes the implications of cycles
        3. It uses the LLM to generate cycle descriptions
        4. It returns properly structured Cycle objects
        """
        # Create the hypergraph analyzer
        analyzer = HypergraphAnalyzer(llm_client=self.mock_llm_client)
        
        # Create a test graph with a cycle
        graph = LegalHypergraph()
        
        # Add nodes
        node1_id = graph.add_node(self.sample_clauses[0])
        node2_id = graph.add_node(self.sample_clauses[1])
        node3_id = graph.add_node(self.sample_clauses[2])
        
        # Add edges to create a cycle
        graph.add_edge(
            edge_type=RelationshipType.REFERENCE.value,
            source_nodes=[node1_id],
            target_nodes=[node3_id],
            edge_data={"reference_text": "Section 3"}
        )
        graph.add_edge(
            edge_type=RelationshipType.REFERENCE.value,
            source_nodes=[node3_id],
            target_nodes=[node2_id],
            edge_data={"reference_text": "Section 2"}
        )
        graph.add_edge(
            edge_type=RelationshipType.REFERENCE.value,
            source_nodes=[node2_id],
            target_nodes=[node1_id],
            edge_data={"reference_text": "Section 1"}
        )
        
        # Configure the mock LLM response for cycle analysis
        cycle_response = """
        Description: Circular reference between payment terms, late fees, and payment schedule
        Severity: HIGH
        Implications:
        - Creates ambiguity in contract execution order
        - May lead to interpretation challenges
        - Could create enforcement difficulties
        """
        self.mock_llm_client.generate.return_value = cycle_response
        
        # Detect cycles
        cycles = analyzer.detect_cycles(graph)
        
        # Verify the LLM was called for cycle analysis
        self.mock_llm_client.generate.assert_called_once()
        
        # Verify the results
        self.assertIsInstance(cycles, list)
        self.assertEqual(len(cycles), 1)
        self.assertIsInstance(cycles[0], Cycle)
        self.assertEqual(cycles[0].severity, "HIGH")
        self.assertEqual(len(cycles[0].implications), 3)
        self.assertIn("circular reference", cycles[0].description.lower())

    def test_analyze_impact(self):
        """
        Test the analyze_impact method.
        
        This test verifies that:
        1. The analyzer correctly identifies direct and indirect impacts
        2. It calculates appropriate risk levels
        3. It handles nodes with various connectivity patterns
        4. It returns properly structured ImpactAnalysis objects
        """
        # Create the hypergraph analyzer
        analyzer = HypergraphAnalyzer(llm_client=self.mock_llm_client)
        
        # Create a test graph with various connections
        graph = LegalHypergraph()
        
        # Add nodes
        node1_id = graph.add_node({"id": "clause1", "text": "Node 1 text"})
        node2_id = graph.add_node({"id": "clause2", "text": "Node 2 text"})
        node3_id = graph.add_node({"id": "clause3", "text": "Node 3 text"})
        node4_id = graph.add_node({"id": "clause4", "text": "Node 4 text"})
        
        # Add edges
        # Node 1 directly impacts Nodes 2 and 3
        graph.add_edge(
            edge_type=RelationshipType.REFERENCE.value,
            source_nodes=[node1_id],
            target_nodes=[node2_id],
            edge_data={"reference_text": "Reference 1-2"}
        )
        graph.add_edge(
            edge_type=RelationshipType.REFERENCE.value,
            source_nodes=[node1_id],
            target_nodes=[node3_id],
            edge_data={"reference_text": "Reference 1-3"}
        )
        
        # Node 3 impacts Node 4 (creating an indirect impact from Node 1 to Node 4)
        graph.add_edge(
            edge_type=RelationshipType.REFERENCE.value,
            source_nodes=[node3_id],
            target_nodes=[node4_id],
            edge_data={"reference_text": "Reference 3-4"}
        )
        
        # Analyze impact of Node 1
        impact = analyzer.analyze_impact(node1_id, graph)
        
        # Verify the results
        self.assertIsInstance(impact, ImpactAnalysis)
        self.assertEqual(impact.node_id, node1_id)
        self.assertEqual(len(impact.direct_impacts), 2)
        self.assertIn(node2_id, impact.direct_impacts)
        self.assertIn(node3_id, impact.direct_impacts)
        self.assertEqual(len(impact.indirect_impacts), 1)
        self.assertIn(node4_id, impact.indirect_impacts)
        
        # Verify risk level calculation
        # 2 direct + 1 indirect should be medium risk
        self.assertEqual(impact.risk_level, "MEDIUM")

    def test_find_critical_nodes(self):
        """
        Test the find_critical_nodes method.
        
        This test verifies that:
        1. The analyzer correctly identifies critical nodes based on connectivity
        2. It calculates appropriate criticality scores
        3. It ranks nodes by criticality
        4. It returns properly structured node information
        """
        # Create the hypergraph analyzer
        analyzer = HypergraphAnalyzer(llm_client=self.mock_llm_client)
        
        # Create a test graph with a critical node
        graph = LegalHypergraph()
        
        # Add nodes
        node1_id = graph.add_node({"id": "clause1", "text": "Critical node text"})
        node2_id = graph.add_node({"id": "clause2", "text": "Node 2 text"})
        node3_id = graph.add_node({"id": "clause3", "text": "Node 3 text"})
        node4_id = graph.add_node({"id": "clause4", "text": "Node 4 text"})
        
        # Add edges to make node1 critical (high connectivity)
        # Incoming edges
        graph.add_edge(
            edge_type=RelationshipType.REFERENCE.value,
            source_nodes=[node2_id],
            target_nodes=[node1_id],
            edge_data={"reference_text": "Reference 2-1"}
        )
        graph.add_edge(
            edge_type=RelationshipType.REFERENCE.value,
            source_nodes=[node3_id],
            target_nodes=[node1_id],
            edge_data={"reference_text": "Reference 3-1"}
        )
        
        # Outgoing edges
        graph.add_edge(
            edge_type=RelationshipType.REFERENCE.value,
            source_nodes=[node1_id],
            target_nodes=[node4_id],
            edge_data={"reference_text": "Reference 1-4"}
        )
        
        # Find critical nodes
        critical_nodes = analyzer.find_critical_nodes(graph)
        
        # Verify the results
        self.assertIsInstance(critical_nodes, list)
        self.assertEqual(len(critical_nodes), 1)  # Only node1 should be critical
        
        critical_node = critical_nodes[0]
        self.assertEqual(critical_node["node_id"], node1_id)
        self.assertGreater(critical_node["criticality_score"], 5)  # Should exceed threshold
        self.assertEqual(critical_node["incoming_connections"], 2)
        self.assertEqual(critical_node["outgoing_connections"], 1)

    def test_analyze_relationship_clusters(self):
        """
        Test the analyze_relationship_clusters method.
        
        This test verifies that:
        1. The analyzer correctly identifies clusters of related clauses
        2. It calculates appropriate cluster metrics (size, density)
        3. It handles disconnected components
        4. It returns properly structured cluster information
        """
        # Create the hypergraph analyzer
        analyzer = HypergraphAnalyzer(llm_client=self.mock_llm_client)
        
        # Create a test graph with two distinct clusters
        graph = LegalHypergraph()
        
        # Add nodes for cluster 1
        c1_node1 = graph.add_node({"id": "c1_clause1", "text": "Cluster 1 node 1"})
        c1_node2 = graph.add_node({"id": "c1_clause2", "text": "Cluster 1 node 2"})
        c1_node3 = graph.add_node({"id": "c1_clause3", "text": "Cluster 1 node 3"})
        
        # Add nodes for cluster 2
        c2_node1 = graph.add_node({"id": "c2_clause1", "text": "Cluster 2 node 1"})
        c2_node2 = graph.add_node({"id": "c2_clause2", "text": "Cluster 2 node 2"})
        
        # Add edges within cluster 1
        graph.add_edge(
            edge_type=RelationshipType.REFERENCE.value,
            source_nodes=[c1_node1],
            target_nodes=[c1_node2],
            edge_data={"reference_text": "Reference c1_1-2"}
        )
        graph.add_edge(
            edge_type=RelationshipType.REFERENCE.value,
            source_nodes=[c1_node2],
            target_nodes=[c1_node3],
            edge_data={"reference_text": "Reference c1_2-3"}
        )
        
        # Add edges within cluster 2
        graph.add_edge(
            edge_type=RelationshipType.REFERENCE.value,
            source_nodes=[c2_node1],
            target_nodes=[c2_node2],
            edge_data={"reference_text": "Reference c2_1-2"}
        )
        
        # Analyze relationship clusters
        clusters = analyzer.analyze_relationship_clusters(graph)
        
        # Verify the results
        self.assertIsInstance(clusters, list)
        self.assertEqual(len(clusters), 2)  # Should find 2 clusters
        
        # Verify cluster 1 (larger cluster)
        cluster1 = next((c for c in clusters if len(c["nodes"]) == 3), None)
        self.assertIsNotNone(cluster1)
        self.assertEqual(cluster1["size"], 3)
        self.assertEqual(len(cluster1["edges"]), 2)
        
        # Verify cluster 2 (smaller cluster)
        cluster2 = next((c for c in clusters if len(c["nodes"]) == 2), None)
        self.assertIsNotNone(cluster2)
        self.assertEqual(cluster2["size"], 2)
        self.assertEqual(len(cluster2["edges"]), 1)


if __name__ == '__main__':
    unittest.main() 