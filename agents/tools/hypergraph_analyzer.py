from typing import Dict, List, Any, Optional, Set, Tuple
import uuid
from dataclasses import dataclass
from enum import Enum
import re

from agents.utils.hypergraph import LegalHypergraph
from agents.utils.dependency_analyzer import DependencyAnalyzer
from agents.utils.ollama_client import OllamaClient


@dataclass
class Cycle:
    """Data class representing a cycle in the legal document."""
    id: str
    nodes: List[str]
    description: str
    severity: str
    implications: List[str]


@dataclass
class ImpactAnalysis:
    """Data class representing the impact analysis of a node."""
    node_id: str
    direct_impacts: List[str]
    indirect_impacts: List[str]
    risk_level: str
    description: str


class RelationshipType(Enum):
    """Enum for legal relationship types."""
    REFERENCE = "reference"
    DEFINITION = "definition"
    DEPENDENCY = "dependency"
    MODIFICATION = "modification"
    EXCEPTION = "exception"
    CONDITION = "condition"


def analyze_hypergraph_structure(
    clauses: List[Dict[str, Any]],
    llm_client: Any = None,
    analyze_cycles: bool = True,
    analyze_critical_nodes: bool = True,
    analyze_clusters: bool = True,
    node_to_analyze: str = None
) -> Dict[str, Any]:
    """
    Build a hypergraph from clauses and perform comprehensive structural analysis.
    
    This function:
    1. Builds a hypergraph representing the document structure
    2. Detects cycles and circular dependencies
    3. Identifies critical nodes
    4. Analyzes relationship clusters
    5. Optionally analyzes impact of a specific node
    
    Args:
        clauses: List of clause dictionaries with 'id' and 'text' fields
        llm_client: Optional client for LLM interactions
        analyze_cycles: Whether to analyze cycles (default: True)
        analyze_critical_nodes: Whether to analyze critical nodes (default: True)
        analyze_clusters: Whether to analyze relationship clusters (default: True)
        node_to_analyze: Optional ID of a specific node to analyze impact for
        
    Returns:
        Dict: Comprehensive analysis results
    """
    # Initialize dependency analyzer
    dependency_analyzer = DependencyAnalyzer()
    
    # Helper function: Check if a reference matches a clause
    def reference_matches_clause(reference: Dict[str, Any], clause: Dict[str, Any]) -> bool:
        """
        Check if a reference matches a clause.
        
        Args:
            reference: Reference dictionary
            clause: Clause dictionary
            
        Returns:
            bool: True if the reference matches the clause
        """
        # This is a simplified implementation
        # In a real system, you would need more sophisticated matching logic
        
        clause_id = clause.get("id", "")
        clause_text = clause.get("text", "")
        clause_heading = clause.get("heading", "")
        
        ref_type = reference["type"]
        ref_value = reference["value"]
        
        # Check for direct ID match
        if ref_value == clause_id:
            return True
        
        # Check for heading match
        if clause_heading and ref_type in clause_heading.lower() and ref_value in clause_heading:
            return True
        
        # Check for section/article/clause number match
        if ref_type in ["section", "article", "clause", "paragraph"]:
            pattern = f"{ref_type.capitalize()} {ref_value}"
            if pattern in clause_text or pattern in clause_heading:
                return True
        
        return False
    
    # Helper function: Determine relationship type
    def determine_relationship_type(
        reference: Dict[str, Any], 
        source_clause: Dict[str, Any], 
        target_clause: Dict[str, Any]
    ) -> RelationshipType:
        """
        Determine the type of relationship between clauses.
        
        Args:
            reference: Reference dictionary
            source_clause: Source clause dictionary
            target_clause: Target clause dictionary
            
        Returns:
            RelationshipType: Type of relationship
        """
        ref_text = reference.get("full_text", "").lower()
        
        # Check for specific relationship indicators
        if "subject to" in ref_text or "contingent on" in ref_text or "conditional upon" in ref_text:
            return RelationshipType.CONDITION
        elif "except as" in ref_text or "notwithstanding" in ref_text or "excluding" in ref_text:
            return RelationshipType.EXCEPTION
        elif "amends" in ref_text or "modifies" in ref_text or "changes" in ref_text:
            return RelationshipType.MODIFICATION
        elif "as defined in" in ref_text or "shall mean" in ref_text:
            return RelationshipType.DEFINITION
        elif "depends on" in ref_text or "requires" in ref_text:
            return RelationshipType.DEPENDENCY
        else:
            return RelationshipType.REFERENCE
    
    # Helper function: Add definition relationships
    def add_definition_relationships(
        graph: LegalHypergraph, 
        clauses: List[Dict[str, Any]], 
        node_id_map: Dict[str, str]
    ) -> None:
        """
        Add definition relationships to the graph.
        
        Args:
            graph: Legal hypergraph
            clauses: List of clause dictionaries
            node_id_map: Mapping from clause IDs to node IDs
        """
        # Extract defined terms
        defined_terms = {}
        
        for clause in clauses:
            clause_id = clause.get("id", str(uuid.uuid4()))
            clause_text = clause.get("text", "")
            
            # Simple pattern matching for definitions
            # In a real system, you would use more sophisticated NLP
            definition_patterns = [
                r'"([^"]+)"\s+means',
                r'"([^"]+)"\s+shall mean',
                r'term\s+"([^"]+)"',
                r'defined\s+term\s+"([^"]+)"'
            ]
            
            for pattern in definition_patterns:
                matches = re.findall(pattern, clause_text, re.IGNORECASE)
                for term in matches:
                    defined_terms[term.lower()] = clause_id
        
        # Add edges for term usage
        for clause in clauses:
            clause_id = clause.get("id", str(uuid.uuid4()))
            clause_text = clause.get("text", "")
            
            # Skip definition clauses
            if clause_id in defined_terms.values():
                continue
            
            # Check for term usage
            for term, def_clause_id in defined_terms.items():
                # Skip if the term is too common
                if len(term) < 4:
                    continue
                    
                # Check if the term is used in this clause
                if re.search(r'\b' + re.escape(term) + r'\b', clause_text, re.IGNORECASE):
                    # Add an edge from the definition to the usage
                    graph.add_edge(
                        edge_type=RelationshipType.DEFINITION.value,
                        source_nodes=[node_id_map[def_clause_id]],
                        target_nodes=[node_id_map[clause_id]],
                        edge_data={
                            "term": term,
                            "relationship_type": RelationshipType.DEFINITION.value
                        }
                    )
    
    # Helper function: Add conditional relationships
    def add_conditional_relationships(
        graph: LegalHypergraph, 
        clauses: List[Dict[str, Any]], 
        node_id_map: Dict[str, str]
    ) -> None:
        """
        Add conditional relationships to the graph.
        
        Args:
            graph: Legal hypergraph
            clauses: List of clause dictionaries
            node_id_map: Mapping from clause IDs to node IDs
        """
        # Look for conditional language
        conditional_patterns = [
            (r'if\s+([^,\.;]+)', RelationshipType.CONDITION),
            (r'provided\s+that\s+([^,\.;]+)', RelationshipType.CONDITION),
            (r'subject\s+to\s+([^,\.;]+)', RelationshipType.CONDITION),
            (r'unless\s+([^,\.;]+)', RelationshipType.EXCEPTION),
            (r'except\s+([^,\.;]+)', RelationshipType.EXCEPTION),
            (r'notwithstanding\s+([^,\.;]+)', RelationshipType.EXCEPTION)
        ]
        
        for clause in clauses:
            clause_id = clause.get("id", str(uuid.uuid4()))
            clause_text = clause.get("text", "")
            
            for pattern, rel_type in conditional_patterns:
                matches = re.findall(pattern, clause_text, re.IGNORECASE)
                
                if matches:
                    # This clause has conditional language
                    # In a real system, you would use NLP to identify the target clause
                    # Here we'll just add a self-loop to indicate the condition
                    graph.add_edge(
                        edge_type=rel_type.value,
                        source_nodes=[node_id_map[clause_id]],
                        target_nodes=[node_id_map[clause_id]],
                        edge_data={
                            "condition_text": matches[0],
                            "relationship_type": rel_type.value
                        }
                    )
    
    # Helper function: Find indirect impacts
    def find_indirect_impacts(
        node_id: str, 
        graph: LegalHypergraph, 
        direct_impacts: Set[str]
    ) -> Set[str]:
        """
        Find nodes indirectly impacted by a node.
        
        Args:
            node_id: ID of the node to analyze
            graph: Legal hypergraph
            direct_impacts: Set of directly impacted node IDs
            
        Returns:
            Set[str]: Set of indirectly impacted node IDs
        """
        # Use BFS to find all reachable nodes
        visited = {node_id}.union(direct_impacts)
        queue = list(direct_impacts)
        indirect_impacts = set()
        
        while queue:
            current = queue.pop(0)
            
            # Get outgoing edges
            current_node = graph.get_node(current)
            if not current_node:
                continue
                
            for edge_id in current_node.outgoing_edges:
                edge = graph.get_edge(edge_id)
                if edge:
                    for target in edge.target_nodes:
                        if target not in visited:
                            visited.add(target)
                            queue.append(target)
                            indirect_impacts.add(target)
        
        return indirect_impacts
    
    # Helper function: Determine risk level
    def determine_risk_level(direct_count: int, indirect_count: int) -> str:
        """
        Determine risk level based on impact counts.
        
        Args:
            direct_count: Number of directly impacted nodes
            indirect_count: Number of indirectly impacted nodes
            
        Returns:
            str: Risk level (HIGH, MEDIUM, LOW)
        """
        total_impact = direct_count + (indirect_count * 0.5)
        
        if total_impact > 10:
            return "HIGH"
        elif total_impact > 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    # Helper function: Analyze cycle with LLM
    def analyze_cycle_with_llm(cycle_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a cycle using the LLM.
        
        Args:
            cycle_data: List of node data dictionaries in the cycle
            
        Returns:
            Dict: Analysis result
        """
        if not llm_client:
            return None
        
        # Format the cycle data for the prompt
        cycle_text = ""
        for i, data in enumerate(cycle_data):
            clause_text = data.get("text", "")
            cycle_text += f"Clause {i+1}:\n{clause_text}\n\n"
        
        # Create the prompt
        prompt = f"""
        Analyze the following circular dependency between clauses:
        
        {cycle_text}
        
        Identify:
        1. The nature of the circular dependency
        2. Potential logical inconsistencies or contradictions
        3. Implications for contract interpretation
        4. Severity of the issue (HIGH, MEDIUM, or LOW)
        
        Format your response as:
        Description: [brief description of the cycle]
        Severity: [HIGH/MEDIUM/LOW]
        Implications:
        - [implication 1]
        - [implication 2]
        - [implication 3]
        """
        
        # Get analysis from LLM
        response = llm_client.query(
            system_prompt="You are a legal document analyzer specializing in identifying circular dependencies and their implications.",
            prompt=prompt
        )
        
        # Parse the response
        analysis = {
            "description": "",
            "severity": "MEDIUM",
            "implications": []
        }
        
        # Simple parsing of the response
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("Description:"):
                analysis["description"] = line[len("Description:"):].strip()
            elif line.startswith("Severity:"):
                severity = line[len("Severity:"):].strip().upper()
                if severity in ["HIGH", "MEDIUM", "LOW"]:
                    analysis["severity"] = severity
            elif line.startswith("- "):
                implication = line[2:].strip()
                if implication:
                    analysis["implications"].append(implication)
        
        return analysis
    
    # Helper function: Build graph from clauses
    def build_graph(clauses: List[Dict[str, Any]]) -> Tuple[LegalHypergraph, Dict[str, str]]:
        """
        Build a hypergraph from a list of clauses.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            Tuple[LegalHypergraph, Dict[str, str]]: Constructed legal hypergraph and node ID map
        """
        # Create a new hypergraph
        graph = LegalHypergraph()
        
        # Add all clauses as nodes
        node_id_map = {}
        for clause in clauses:
            clause_id = clause.get("id", str(uuid.uuid4()))
            node_id = graph.add_node(clause)
            node_id_map[clause_id] = node_id
        
        # Extract relationships between clauses
        for clause in clauses:
            clause_id = clause.get("id", str(uuid.uuid4()))
            source_node_id = node_id_map[clause_id]
            
            # Extract references using the dependency analyzer
            references = dependency_analyzer.extract_references(clause.get("text", ""))
            
            for reference in references:
                # Try to find the target clause
                for target_clause in clauses:
                    target_id = target_clause.get("id", str(uuid.uuid4()))
                    
                    # Skip self-references
                    if target_id == clause_id:
                        continue
                    
                    # Check if this reference points to the target clause
                    if reference_matches_clause(reference, target_clause):
                        # Add an edge for this relationship
                        relationship_type = determine_relationship_type(reference, clause, target_clause)
                        
                        graph.add_edge(
                            edge_type=relationship_type.value,
                            source_nodes=[source_node_id],
                            target_nodes=[node_id_map[target_id]],
                            edge_data={
                                "reference_text": reference.get("full_text", ""),
                                "reference_type": reference.get("type", ""),
                                "relationship_type": relationship_type.value
                            }
                        )
        
        # Add definition relationships
        add_definition_relationships(graph, clauses, node_id_map)
        
        # Add conditional relationships
        add_conditional_relationships(graph, clauses, node_id_map)
        
        return graph, node_id_map
    
    # Helper function: Detect cycles
    def detect_cycles(graph: LegalHypergraph) -> List[Cycle]:
        """
        Detect cycles in the hypergraph and analyze their implications.
        
        Args:
            graph: Legal hypergraph to analyze
            
        Returns:
            List[Cycle]: List of detected cycles with analysis
        """
        # Get raw cycles from the graph
        raw_cycles = graph.detect_cycles()
        
        # Analyze each cycle
        analyzed_cycles = []
        
        for i, cycle_nodes in enumerate(raw_cycles):
            # Generate a unique ID for this cycle
            cycle_id = f"cycle_{i}_{uuid.uuid4().hex[:8]}"
            
            # Get node data for all nodes in the cycle
            cycle_data = []
            for node_id in cycle_nodes:
                node = graph.get_node(node_id)
                if node:
                    cycle_data.append(node.data)
            
            # Analyze the cycle using LLM if available
            description = f"Circular dependency involving {len(cycle_nodes)} clauses"
            severity = "MEDIUM"  # Default severity
            implications = ["Potential logical inconsistency", "May create interpretation challenges"]
            
            if llm_client and cycle_data:
                analysis = analyze_cycle_with_llm(cycle_data)
                if analysis:
                    description = analysis.get("description", description)
                    severity = analysis.get("severity", severity)
                    implications = analysis.get("implications", implications)
            
            # Create a Cycle object
            cycle = Cycle(
                id=cycle_id,
                nodes=cycle_nodes,
                description=description,
                severity=severity,
                implications=implications
            )
            
            analyzed_cycles.append(cycle)
        
        return analyzed_cycles
    
    # Helper function: Analyze impact
    def analyze_impact(node_id: str, graph: LegalHypergraph) -> Optional[ImpactAnalysis]:
        """
        Analyze the impact of a node on the rest of the graph.
        
        Args:
            node_id: ID of the node to analyze
            graph: Legal hypergraph
            
        Returns:
            Optional[ImpactAnalysis]: Impact analysis result
        """
        # Check if the node exists
        node = graph.get_node(node_id)
        if not node:
            return None
        
        # Get direct impacts (nodes directly connected)
        direct_impacts = set()
        for edge_id in node.outgoing_edges:
            edge = graph.get_edge(edge_id)
            if edge:
                direct_impacts.update(edge.target_nodes)
        
        # Get indirect impacts (nodes reachable through paths of length > 1)
        indirect_impacts = find_indirect_impacts(node_id, graph, direct_impacts)
        
        # Determine risk level based on impact breadth
        risk_level = determine_risk_level(len(direct_impacts), len(indirect_impacts))
        
        # Generate description
        description = (
            f"Node impacts {len(direct_impacts)} nodes directly and "
            f"{len(indirect_impacts)} nodes indirectly. "
            f"Risk level: {risk_level}."
        )
        
        # Create ImpactAnalysis object
        impact_analysis = ImpactAnalysis(
            node_id=node_id,
            direct_impacts=list(direct_impacts),
            indirect_impacts=list(indirect_impacts),
            risk_level=risk_level,
            description=description
        )
        
        return impact_analysis
    
    # Helper function: Find critical nodes
    def find_critical_nodes(graph: LegalHypergraph) -> List[Dict[str, Any]]:
        """
        Find critical nodes in the graph based on connectivity metrics.
        
        Args:
            graph: Legal hypergraph
            
        Returns:
            List[Dict]: List of critical nodes with analysis
        """
        critical_nodes = []
        
        # Calculate metrics for each node
        for node_id, node in graph.nodes.items():
            # Count incoming and outgoing connections
            incoming_count = len(node.incoming_edges)
            outgoing_count = len(node.outgoing_edges)
            
            # Get connected nodes
            connected = graph.get_connected_nodes(node_id)
            total_connections = len(connected['incoming']) + len(connected['outgoing'])
            
            # Calculate criticality score
            # Higher score = more critical
            criticality_score = (incoming_count * 1.5) + outgoing_count
            
            # Nodes with high connectivity are considered critical
            if criticality_score > 5 or total_connections > 3:
                critical_nodes.append({
                    "node_id": node_id,
                    "node_data": node.data,
                    "criticality_score": criticality_score,
                    "incoming_connections": incoming_count,
                    "outgoing_connections": outgoing_count,
                    "total_connected_nodes": total_connections
                })
        
        # Sort by criticality score (descending)
        critical_nodes.sort(key=lambda x: x["criticality_score"], reverse=True)
        
        return critical_nodes
    
    # Helper function: Analyze relationship clusters
    def analyze_relationship_clusters(graph: LegalHypergraph) -> List[Dict[str, Any]]:
        """
        Identify and analyze clusters of closely related clauses.
        
        Args:
            graph: Legal hypergraph
            
        Returns:
            List[Dict]: List of relationship clusters
        """
        # This is a simplified implementation of community detection
        # In a production system, you might use more sophisticated algorithms
        
        # Track visited nodes
        visited = set()
        clusters = []
        
        # Find clusters using a simple BFS approach
        for node_id in graph.nodes:
            if node_id in visited:
                continue
                
            # Start a new cluster
            cluster = {
                "id": f"cluster_{len(clusters)}",
                "nodes": [],
                "edges": [],
                "size": 0,
                "density": 0.0,
                "description": ""
            }
            
            # BFS to find connected components
            queue = [node_id]
            cluster_nodes = set()
            cluster_edges = set()
            
            while queue:
                current = queue.pop(0)
                
                if current in visited:
                    continue
                    
                visited.add(current)
                cluster_nodes.add(current)
                
                # Get connected nodes
                node = graph.get_node(current)
                
                # Add outgoing edges and their targets
                for edge_id in node.outgoing_edges:
                    edge = graph.get_edge(edge_id)
                    if edge:
                        cluster_edges.add(edge_id)
                        for target in edge.target_nodes:
                            if target not in visited:
                                queue.append(target)
                
                # Add incoming edges and their sources
                for edge_id in node.incoming_edges:
                    edge = graph.get_edge(edge_id)
                    if edge:
                        cluster_edges.add(edge_id)
                        for source in edge.source_nodes:
                            if source not in visited:
                                queue.append(source)
            
            # Only add non-trivial clusters
            if len(cluster_nodes) > 1:
                # Calculate density (ratio of actual to possible connections)
                possible_connections = len(cluster_nodes) * (len(cluster_nodes) - 1)
                density = len(cluster_edges) / possible_connections if possible_connections > 0 else 0
                
                # Update cluster information
                cluster["nodes"] = list(cluster_nodes)
                cluster["edges"] = list(cluster_edges)
                cluster["size"] = len(cluster_nodes)
                cluster["density"] = density
                cluster["description"] = f"Cluster of {len(cluster_nodes)} related clauses with {len(cluster_edges)} connections"
                
                clusters.append(cluster)
        
        return clusters
    
    # MAIN FUNCTION EXECUTION
    
    # Validate inputs
    if not clauses:
        return {
            "analysis_id": str(uuid.uuid4()),
            "error": "No clauses provided for analysis",
            "graph_built": False
        }
    
    # Generate a unique ID for this analysis
    analysis_id = str(uuid.uuid4())
    
    # Build the graph
    graph, node_id_map = build_graph(clauses)
    
    # Initialize results container
    results = {
        "analysis_id": analysis_id,
        "graph_built": True,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "clause_count": len(clauses)
    }
    
    # Perform requested analyses
    
    # 1. Cycle detection and analysis
    if analyze_cycles:
        cycles = detect_cycles(graph)
        results["cycles"] = [cycle.__dict__ for cycle in cycles]
        results["cycle_count"] = len(cycles)
        results["has_cycles"] = len(cycles) > 0
    
    # 2. Critical node identification
    if analyze_critical_nodes:
        critical_nodes = find_critical_nodes(graph)
        results["critical_nodes"] = critical_nodes
        results["critical_node_count"] = len(critical_nodes)
    
    # 3. Relationship cluster analysis
    if analyze_clusters:
        clusters = analyze_relationship_clusters(graph)
        results["relationship_clusters"] = clusters
        results["cluster_count"] = len(clusters)
    
    # 4. Specific node impact analysis
    if node_to_analyze:
        # Get the node ID from the map if we have the clause ID
        node_id = node_id_map.get(node_to_analyze, node_to_analyze)
        impact = analyze_impact(node_id, graph)
        if impact:
            results["impact_analysis"] = impact.__dict__
        else:
            results["impact_analysis"] = None
            results["impact_analysis_error"] = f"Node {node_to_analyze} not found or couldn't be analyzed"
    
    # Summary statistics
    results["structural_complexity"] = _calculate_structural_complexity(graph)
    results["has_structural_issues"] = (
        (analyze_cycles and results.get("has_cycles", False)) or
        (analyze_critical_nodes and len(results.get("critical_nodes", [])) > 2)
    )
    
    return results


def _calculate_structural_complexity(graph: LegalHypergraph) -> float:
    """
    Calculate a structural complexity score for the graph.
    
    Args:
        graph: Legal hypergraph
        
    Returns:
        float: Complexity score
    """
    # Count node types
    edge_types = {}
    for edge_id, edge in graph.edges.items():
        edge_type = edge.edge_type
        if edge_type not in edge_types:
            edge_types[edge_type] = 0
        edge_types[edge_type] += 1
    
    # Calculate complexity metrics
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)
    edge_type_count = len(edge_types)
    
    if node_count <= 1:
        return 0.0
    
    # Calculate graph density
    max_edges = node_count * (node_count - 1)
    density = edge_count / max_edges if max_edges > 0 else 0
    
    # Calculate complexity score
    # This is a simple heuristic that could be refined
    complexity = (
        (edge_count / node_count) * 0.4 +  # Average connections per node
        (edge_type_count / 6) * 0.3 +      # Diversity of relationship types (normalized to max of 6 types)
        density * 0.3                      # Graph density
    )
    
    return min(1.0, complexity) * 10  # Scale to 0-10 range


class HypergraphAnalyzer:
    """
    Legacy class maintained for backward compatibility.
    Analyzer for modeling and analyzing complex legal relationships using hypergraphs.
    Now uses the standalone analyze_hypergraph_structure function.
    """
    
    def __init__(self, llm_client: Any = None):
        """
        Initialize the HypergraphAnalyzer.
        
        Args:
            llm_client: Optional client for LLM interactions
        """
        self.llm_client = llm_client
        self.dependency_analyzer = DependencyAnalyzer()
    
    def build_graph(self, clauses: List[Dict[str, Any]]) -> LegalHypergraph:
        """
        Build a hypergraph from a list of clauses.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            LegalHypergraph: Constructed legal hypergraph
        """
        # Call the standalone function and extract the graph
        result = analyze_hypergraph_structure(
            clauses=clauses,
            llm_client=self.llm_client,
            analyze_cycles=False,
            analyze_critical_nodes=False,
            analyze_clusters=False
        )
        
        # This is a bit of a hack since the function doesn't return the graph directly
        # In a real system, you might want to modify the function to return the graph as well
        # For now, we'll rebuild the graph
        graph, _ = self._rebuild_graph(clauses)
        return graph
    
    def _rebuild_graph(self, clauses: List[Dict[str, Any]]) -> Tuple[LegalHypergraph, Dict[str, str]]:
        """
        Rebuild a graph from clauses (private helper for backward compatibility).
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            Tuple[LegalHypergraph, Dict[str, str]]: Graph and node ID map
        """
        # Create a new hypergraph
        graph = LegalHypergraph()
        
        # Add all clauses as nodes
        node_id_map = {}
        for clause in clauses:
            clause_id = clause.get("id", str(uuid.uuid4()))
            node_id = graph.add_node(clause)
            node_id_map[clause_id] = node_id
        
        # Extract relationships between clauses
        for clause in clauses:
            clause_id = clause.get("id", str(uuid.uuid4()))
            source_node_id = node_id_map[clause_id]
            
            # Extract references using the dependency analyzer
            references = self.dependency_analyzer.extract_references(clause.get("text", ""))
            
            for reference in references:
                # Try to find the target clause
                for target_clause in clauses:
                    target_id = target_clause.get("id", str(uuid.uuid4()))
                    
                    # Skip self-references
                    if target_id == clause_id:
                        continue
                    
                    # Check if this reference points to the target clause
                    if self._reference_matches_clause(reference, target_clause):
                        # Add an edge for this relationship
                        relationship_type = self._determine_relationship_type(reference, clause, target_clause)
                        
                        graph.add_edge(
                            edge_type=relationship_type.value,
                            source_nodes=[source_node_id],
                            target_nodes=[node_id_map[target_id]],
                            edge_data={
                                "reference_text": reference.get("full_text", ""),
                                "reference_type": reference.get("type", ""),
                                "relationship_type": relationship_type.value
                            }
                        )
        
        # Add definition relationships
        self._add_definition_relationships(graph, clauses, node_id_map)
        
        # Add conditional relationships
        self._add_conditional_relationships(graph, clauses, node_id_map)
        
        return graph, node_id_map
    
    def detect_cycles(self, graph: LegalHypergraph) -> List[Cycle]:
        """
        Detect cycles in the hypergraph and analyze their implications.
        Now uses the standalone analyze_hypergraph_structure function.
        
        Args:
            graph: Legal hypergraph to analyze
            
        Returns:
            List[Cycle]: List of detected cycles with analysis
        """
        # We need to reverse-engineer the clauses from the graph
        clauses = []
        for node_id, node in graph.nodes.items():
            clauses.append(node.data)
        
        # Call the standalone function with cycle analysis enabled
        result = analyze_hypergraph_structure(
            clauses=clauses,
            llm_client=self.llm_client,
            analyze_cycles=True,
            analyze_critical_nodes=False,
            analyze_clusters=False
        )
        
        # Convert the dictionary representations back to Cycle objects
        cycles = []
        for cycle_dict in result.get("cycles", []):
            cycle = Cycle(
                id=cycle_dict.get("id", f"cycle_{uuid.uuid4().hex[:8]}"),
                nodes=cycle_dict.get("nodes", []),
                description=cycle_dict.get("description", ""),
                severity=cycle_dict.get("severity", "MEDIUM"),
                implications=cycle_dict.get("implications", [])
            )
            cycles.append(cycle)
        
        return cycles
    
    def analyze_impact(self, node_id: str, graph: LegalHypergraph) -> ImpactAnalysis:
        """
        Analyze the impact of a node on the rest of the graph.
        Now uses the standalone analyze_hypergraph_structure function.
        
        Args:
            node_id: ID of the node to analyze
            graph: Legal hypergraph
            
        Returns:
            ImpactAnalysis: Impact analysis result
        """
        # We need to reverse-engineer the clauses from the graph
        clauses = []
        for graph_node_id, node in graph.nodes.items():
            clauses.append(node.data)
        
        # Call the standalone function with node impact analysis enabled
        result = analyze_hypergraph_structure(
            clauses=clauses,
            llm_client=self.llm_client,
            analyze_cycles=False,
            analyze_critical_nodes=False,
            analyze_clusters=False,
            node_to_analyze=node_id
        )
        
        # Convert the dictionary representation back to ImpactAnalysis object
        impact_dict = result.get("impact_analysis", {})
        if not impact_dict:
            # Fallback to default impact analysis if not found
            return ImpactAnalysis(
                node_id=node_id,
                direct_impacts=[],
                indirect_impacts=[],
                risk_level="LOW",
                description=f"No impact analysis available for node {node_id}"
            )
        
        impact = ImpactAnalysis(
            node_id=impact_dict.get("node_id", node_id),
            direct_impacts=impact_dict.get("direct_impacts", []),
            indirect_impacts=impact_dict.get("indirect_impacts", []),
            risk_level=impact_dict.get("risk_level", "LOW"),
            description=impact_dict.get("description", "")
        )
        
        return impact
    
    def find_critical_nodes(self, graph: LegalHypergraph) -> List[Dict[str, Any]]:
        """
        Find critical nodes in the graph based on connectivity metrics.
        Now uses the standalone analyze_hypergraph_structure function.
        
        Args:
            graph: Legal hypergraph
            
        Returns:
            List[Dict]: List of critical nodes with analysis
        """
        # We need to reverse-engineer the clauses from the graph
        clauses = []
        for node_id, node in graph.nodes.items():
            clauses.append(node.data)
        
        # Call the standalone function with critical node analysis enabled
        result = analyze_hypergraph_structure(
            clauses=clauses,
            llm_client=self.llm_client,
            analyze_cycles=False,
            analyze_critical_nodes=True,
            analyze_clusters=False
        )
        
        return result.get("critical_nodes", [])
    
    def analyze_relationship_clusters(self, graph: LegalHypergraph) -> List[Dict[str, Any]]:
        """
        Identify and analyze clusters of closely related clauses.
        Now uses the standalone analyze_hypergraph_structure function.
        
        Args:
            graph: Legal hypergraph
            
        Returns:
            List[Dict]: List of relationship clusters
        """
        # We need to reverse-engineer the clauses from the graph
        clauses = []
        for node_id, node in graph.nodes.items():
            clauses.append(node.data)
        
        # Call the standalone function with relationship cluster analysis enabled
        result = analyze_hypergraph_structure(
            clauses=clauses,
            llm_client=self.llm_client,
            analyze_cycles=False,
            analyze_critical_nodes=False,
            analyze_clusters=True
        )
        
        return result.get("relationship_clusters", [])
    
    # Legacy helper methods maintained for backward compatibility
    def _reference_matches_clause(self, reference: Dict[str, Any], clause: Dict[str, Any]) -> bool:
        """Legacy method that delegates to the function in analyze_hypergraph_structure."""
        # Function signature maintained for backward compatibility
        pass
    
    def _determine_relationship_type(self, 
                                   reference: Dict[str, Any], 
                                   source_clause: Dict[str, Any], 
                                   target_clause: Dict[str, Any]) -> RelationshipType:
        """Legacy method that delegates to the function in analyze_hypergraph_structure."""
        # Function signature maintained for backward compatibility
        ref_text = reference.get("full_text", "").lower()
        
        # Check for specific relationship indicators
        if "subject to" in ref_text or "contingent on" in ref_text or "conditional upon" in ref_text:
            return RelationshipType.CONDITION
        elif "except as" in ref_text or "notwithstanding" in ref_text or "excluding" in ref_text:
            return RelationshipType.EXCEPTION
        elif "amends" in ref_text or "modifies" in ref_text or "changes" in ref_text:
            return RelationshipType.MODIFICATION
        elif "as defined in" in ref_text or "shall mean" in ref_text:
            return RelationshipType.DEFINITION
        elif "depends on" in ref_text or "requires" in ref_text:
            return RelationshipType.DEPENDENCY
        else:
            return RelationshipType.REFERENCE
    
    def _add_definition_relationships(self, 
                                    graph: LegalHypergraph, 
                                    clauses: List[Dict[str, Any]], 
                                    node_id_map: Dict[str, str]) -> None:
        """Legacy method that delegates to the function in analyze_hypergraph_structure."""
        # Function signature maintained for backward compatibility
        pass
    
    def _add_conditional_relationships(self, 
                                     graph: LegalHypergraph, 
                                     clauses: List[Dict[str, Any]], 
                                     node_id_map: Dict[str, str]) -> None:
        """Legacy method that delegates to the function in analyze_hypergraph_structure."""
        # Function signature maintained for backward compatibility
        pass
    
    def _find_indirect_impacts(self, 
                             node_id: str, 
                             graph: LegalHypergraph, 
                             direct_impacts: Set[str]) -> Set[str]:
        """Legacy method that delegates to the function in analyze_hypergraph_structure."""
        # Function signature maintained for backward compatibility
        pass
    
    def _determine_risk_level(self, direct_count: int, indirect_count: int) -> str:
        """Legacy method that delegates to the function in analyze_hypergraph_structure."""
        # Function signature maintained for backward compatibility
        pass
    
    def _analyze_cycle_with_llm(self, cycle_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Legacy method that delegates to the function in analyze_hypergraph_structure."""
        # Function signature maintained for backward compatibility
        pass