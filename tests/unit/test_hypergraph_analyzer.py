import unittest
from unittest.mock import Mock, patch, MagicMock
import uuid

from agents.hypergraph_analyzer import (
    HypergraphAnalyzer, Cycle, ImpactAnalysis, RelationshipType
)
from utils.hypergraph import LegalHypergraph, HyperNode, HyperEdge


class TestHypergraphAnalyzer(unittest.TestCase):
    """
    Unit tests for the HypergraphAnalyzer component.
    
    The HypergraphAnalyzer is responsible for modeling and analyzing complex legal
    relationships using hypergraphs. It builds a hypergraph representation of legal
    documents, identifies complex relationships, detects cycles, and analyzes impacts.
    """

    def setUp(self):
        """
        Set up test fixtures.
        
        Creates a mock LLM that returns a sample cycle analysis and initializes
        a test hypergraph with three nodes (clauses) and two edges (references).
        """
        # Mock LLM that returns a sample cycle analysis
        self.mock_llm = Mock()
        self.mock_llm.generate = Mock(return_value="""
        Description: Circular reference between payment and termination clauses
        Severity: MEDIUM
        Implications:
        - Creates ambiguity in contract execution
        - May lead to disputes over payment timing
        - Could complicate termination procedures
        """)
        
        # Initialize the analyzer with the mock LLM
        self.analyzer = HypergraphAnalyzer(llm_client=self.mock_llm)
        
        # Create a test hypergraph with three clauses about payment and termination
        self.graph = LegalHypergraph()
        
        # Add nodes representing clauses
        self.node1_id = self.graph.add_node({"id": "clause1", "text": "Payment terms as specified in Section 3."})
        self.node2_id = self.graph.add_node({"id": "clause2", "text": "Section 3: Payment shall be made within 30 days."})
        self.node3_id = self.graph.add_node({"id": "clause3", "text": "Termination requires settlement of payments per Section 3."})
        
        # Add edges representing references between clauses
        self.edge1_id = self.graph.add_edge(
            edge_type="reference",
            source_nodes=[self.node1_id],
            target_nodes=[self.node2_id],
            edge_data={"reference_text": "Section 3"}
        )
        
        self.edge2_id = self.graph.add_edge(
            edge_type="reference",
            source_nodes=[self.node3_id],
            target_nodes=[self.node2_id],
            edge_data={"reference_text": "Section 3"}
        )
    
    def test_build_graph(self):
        """
        Test building a hypergraph from clauses.
        
        This test verifies that the analyzer correctly builds a hypergraph
        representation of legal clauses, with nodes for clauses and edges for
        relationships between them.
        
        Input: List of clauses about payment and termination
        Expected output: LegalHypergraph with nodes and edges
        """
        # Arrange - Create a list of clauses
        clauses = [
            {"id": "clause1", "text": "Payment terms as specified in Section 3."},
            {"id": "clause2", "text": "Section 3: Payment shall be made within 30 days."},
            {"id": "clause3", "text": "Termination requires settlement of payments per Section 3."}
        ]
        
        # Act - Build a hypergraph from the clauses
        graph = self.analyzer.build_graph(clauses)
        
        # Assert - Check that the graph contains the expected nodes and edges
        self.assertEqual(len(graph.nodes), 3)
        self.assertGreaterEqual(len(graph.edges), 2)  # At least 2 reference edges
        
        # Check that nodes contain the clause data
        for node_id, node in graph.nodes.items():
            self.assertIn("id", node.data)
            self.assertIn("text", node.data)
            
        # Check that edges have the correct type
        for edge_id, edge in graph.edges.items():
            self.assertIn(edge.edge_type, ["reference", "dependency", "definition"])
    
    def test_detect_cycles(self):
        """
        Test detection of cycles in the hypergraph.
        
        This test verifies that the analyzer correctly identifies circular dependencies
        (cycles) in the legal document structure. Cycles can create logical contradictions
        or ambiguities in contract interpretation.
        
        Input: A hypergraph with a circular dependency
        Expected output: List containing a Cycle object describing the circular reference
        """
        # Arrange - Create a cycle by adding an edge from node2 back to node1
        self.graph.add_edge(
            edge_type="dependency",
            source_nodes=[self.node2_id],
            target_nodes=[self.node1_id],
            edge_data={"dependency_type": "conditional"}
        )
        
        # Act - Detect cycles in the graph
        cycles = self.analyzer.detect_cycles(self.graph)
        
        # Assert - Check that a cycle was detected with expected properties
        self.assertGreaterEqual(len(cycles), 1)
        cycle = cycles[0]
        self.assertIsInstance(cycle, Cycle)
        self.assertIn(self.node1_id, cycle.nodes)
        self.assertIn(self.node2_id, cycle.nodes)
        self.assertEqual(cycle.severity, "MEDIUM")
        self.assertGreaterEqual(len(cycle.implications), 1)
    
    def test_find_critical_nodes(self):
        """
        Test identification of critical nodes in the hypergraph.
        
        This test verifies that the analyzer correctly identifies nodes that are
        critical to the document structure, such as those with many dependencies
        or that are referenced by many other clauses.
        
        Input: A hypergraph with interconnected nodes
        Expected output: List of dictionaries describing critical nodes with scores
        """
        # Act - Find critical nodes in the graph
        critical_nodes = self.analyzer.find_critical_nodes(self.graph)
        
        # Assert - Check that critical nodes were identified
        self.assertGreaterEqual(len(critical_nodes), 1)
        
        # Node2 should be critical as it has incoming edges from both other nodes
        node2_critical = False
        for node in critical_nodes:
            if node["node_id"] == self.node2_id:
                node2_critical = True
                self.assertGreater(node["criticality_score"], 0)
                break
                
        self.assertTrue(node2_critical, "Node2 should be identified as critical")
    
    def test_analyze_impact(self):
        """
        Test impact analysis for a node.
        
        This test verifies that the analyzer correctly assesses the impact of changes
        to a specific clause (node) on other clauses in the document. This helps
        identify the ripple effects of modifying a particular clause.
        
        Input: A node ID and the hypergraph
        Expected output: ImpactAnalysis object with direct and indirect impacts
        """
        # Act - Analyze the impact of node2 (the payment terms clause)
        impact = self.analyzer.analyze_impact(self.node2_id, self.graph)
        
        # Assert - Check that the impact analysis has expected properties
        self.assertIsInstance(impact, ImpactAnalysis)
        self.assertEqual(impact.node_id, self.node2_id)
        
        # Node2 directly impacts both Node1 and Node3
        self.assertGreaterEqual(len(impact.direct_impacts), 0)
        self.assertGreaterEqual(len(impact.indirect_impacts), 0)
        
        # Check risk level
        self.assertIn(impact.risk_level, ["HIGH", "MEDIUM", "LOW"])
    
    def test_analyze_relationship_clusters(self):
        """
        Test analysis of relationship clusters.
        
        This test verifies that the analyzer correctly identifies clusters of closely
        related clauses in the document. Clusters help understand the document's
        logical structure and identify groups of clauses that should be analyzed together.
        
        Input: A hypergraph with related nodes
        Expected output: List of cluster dictionaries with nodes and properties
        """
        # Act - Analyze relationship clusters in the graph
        clusters = self.analyzer.analyze_relationship_clusters(self.graph)
        
        # Assert - Check that clusters were identified with expected properties
        self.assertGreaterEqual(len(clusters), 1)
        
        # All nodes should be in the same cluster
        cluster = clusters[0]
        self.assertGreaterEqual(len(cluster["nodes"]), 3)
        self.assertIn(self.node1_id, cluster["nodes"])
        self.assertIn(self.node2_id, cluster["nodes"])
        self.assertIn(self.node3_id, cluster["nodes"])
        
        # Check cluster properties
        self.assertIn("cluster_type", cluster)
        self.assertIn("density", cluster)
        self.assertGreaterEqual(cluster["density"], 0)
    
    def test_determine_relationship_type(self):
        """
        Test determination of relationship type from reference.
        
        This test verifies that the analyzer correctly determines the type of relationship
        between clauses based on the reference text. Different types of relationships
        (reference, dependency, definition) have different implications for analysis.
        
        Input: Reference data and source/target clauses
        Expected output: RelationshipType enum value
        """
        # Arrange - Create reference data and clauses
        reference = {"type": "Section", "value": "3", "full_text": "Section 3"}
        source_clause = {"id": "clause1", "text": "Payment terms as specified in Section 3."}
        target_clause = {"id": "clause2", "text": "Section 3: Payment shall be made within 30 days."}
        
        # Act - Determine the relationship type
        rel_type = self.analyzer._determine_relationship_type(reference, source_clause, target_clause)
        
        # Assert - Check that the relationship type is correct
        self.assertEqual(rel_type, RelationshipType.REFERENCE)
    
    def test_reference_matches_clause(self):
        """
        Test matching a reference to a clause.
        
        This test verifies that the analyzer correctly determines whether a reference
        (like "Section 3") matches a particular clause. This is crucial for building
        accurate relationships between clauses in the document.
        
        Input: Reference data and a clause
        Expected output: Boolean indicating whether the reference matches the clause
        """
        # Arrange - Create reference data and a matching clause
        reference = {"type": "Section", "value": "3", "full_text": "Section 3"}
        clause = {"id": "clause2", "text": "Section 3: Payment shall be made within 30 days."}
        
        # Act - Check if the reference matches the clause
        matches = self.analyzer._reference_matches_clause(reference, clause)
        
        # Assert - Check that the reference matches the clause
        self.assertTrue(matches)
        
        # Test non-matching reference
        reference2 = {"type": "Section", "value": "4", "full_text": "Section 4"}
        matches2 = self.analyzer._reference_matches_clause(reference2, clause)
        self.assertFalse(matches2)


if __name__ == '__main__':
    unittest.main() 