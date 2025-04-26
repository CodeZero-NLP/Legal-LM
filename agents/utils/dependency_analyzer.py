from typing import Dict, List, Any, Set, Tuple, Optional
from .hypergraph import LegalHypergraph
import re

class DependencyAnalyzer:
    """
    Analyzer for identifying and resolving long-range dependencies in legal documents.
    """
    
    def __init__(self):
        """Initialize the dependency analyzer."""
        # Patterns for identifying common dependency references
        self.reference_patterns = [
            r"Section\s+(\d+(?:\.\d+)*)",  # Section references
            r"Article\s+(\d+(?:\.\d+)*)",  # Article references
            r"Clause\s+(\d+(?:\.\d+)*)",   # Clause references
            r"Paragraph\s+(\d+(?:\.\d+)*)", # Paragraph references
            r"as defined in\s+(\w+)",      # Definition references
            r"pursuant to\s+(\w+)",        # Pursuant references
            r"in accordance with\s+(\w+)", # Accordance references
            r"subject to\s+(\w+)"          # Subject to references
        ]
        
        # Compile regex patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.reference_patterns]
    
    def extract_references(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract references from text using regex patterns.
        
        Args:
            text: Text to analyze for references
            
        Returns:
            List[Dict]: List of found references
        """
        references = []
        
        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.finditer(text)
            for match in matches:
                ref_type = self.reference_patterns[i].split(r'\s+')[0].lower()
                ref_value = match.group(1)
                references.append({
                    "type": ref_type,
                    "value": ref_value,
                    "start": match.start(),
                    "end": match.end(),
                    "full_text": match.group(0)
                })
        
        return references
    
    def build_dependency_graph(self, clauses: List[Dict[str, Any]]) -> LegalHypergraph:
        """
        Build a dependency graph from a list of clauses.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            LegalHypergraph: Hypergraph representing dependencies
        """
        graph = LegalHypergraph()
        
        # Add all clauses as nodes first
        node_id_map = {}
        for clause in clauses:
            clause_id = clause.get("id")
            node_id = graph.add_node(clause)
            node_id_map[clause_id] = node_id
        
        # Now add edges for dependencies
        for clause in clauses:
            clause_id = clause.get("id")
            clause_text = clause.get("text", "")
            
            # Extract references from this clause
            references = self.extract_references(clause_text)
            
            # For each reference, try to find the target clause
            for ref in references:
                # Logic to match reference to target clauses
                # This is a simplified approach - in a real system, you'd need more sophisticated matching
                for target_clause in clauses:
                    target_id = target_clause.get("id")
                    if target_id == clause_id:  # Skip self-references
                        continue
                        
                    # Simple matching - in reality, you'd need context-aware matching
                    if (ref["type"] == "section" and f"Section {ref['value']}" in target_clause.get("heading", "")) or \
                       (ref["type"] == "article" and f"Article {ref['value']}" in target_clause.get("heading", "")) or \
                       (ref["type"] == "clause" and f"Clause {ref['value']}" in target_clause.get("heading", "")):
                        # Add dependency edge
                        graph.add_edge(
                            edge_type="reference",
                            source_nodes=[node_id_map[clause_id]],
                            target_nodes=[node_id_map[target_id]],
                            edge_data={"reference_type": ref["type"]}
                        )
        
        return graph
    
    def find_long_range_dependencies(self, graph: LegalHypergraph, min_distance: int = 5) -> List[Dict[str, Any]]:
        """
        Find long-range dependencies in the graph.
        
        Args:
            graph: Dependency hypergraph
            min_distance: Minimum distance to consider a dependency 'long-range'
            
        Returns:
            List[Dict]: List of long-range dependencies
        """
        long_range_deps = []
        
        # For each node, find path lengths to all other connected nodes
        for source_id, source_node in graph.nodes.items():
            # BFS to find distances
            queue = [(source_id, 0)]  # (node_id, distance)
            visited = {source_id}
            
            while queue:
                current_id, distance = queue.pop(0)
                current_node = graph.nodes[current_id]
                
                # Process outgoing edges
                for edge_id in current_node.outgoing_edges:
                    edge = graph.edges[edge_id]
                    for target_id in edge.target_nodes:
                        if target_id not in visited:
                            visited.add(target_id)
                            queue.append((target_id, distance + 1))
                            
                            # If the distance is >= min_distance, it's a long-range dependency
                            if distance + 1 >= min_distance:
                                long_range_deps.append({
                                    "source": source_id,
                                    "target": target_id,
                                    "distance": distance + 1,
                                    "edge_type": edge.data.get("type", "unknown"),
                                    "source_data": graph.nodes[source_id].data,
                                    "target_data": graph.nodes[target_id].data
                                })
        
        return long_range_deps 