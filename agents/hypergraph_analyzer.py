from typing import Dict, List, Any, Optional, Set, Tuple
import uuid
from dataclasses import dataclass
from enum import Enum

from utils.hypergraph import LegalHypergraph
from utils.dependency_analyzer import DependencyAnalyzer
from utils.ollama_client import OllamaClient


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


class HypergraphAnalyzer:
    """
    Analyzer for modeling and analyzing complex legal relationships using hypergraphs.
    
    This component builds a hypergraph representation of legal documents,
    identifies complex relationships, detects cycles, and analyzes impacts.
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
        
        return graph
    
    def detect_cycles(self, graph: LegalHypergraph) -> List[Cycle]:
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
            
            if self.llm_client and cycle_data:
                analysis = self._analyze_cycle_with_llm(cycle_data)
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
    
    def analyze_impact(self, node_id: str, graph: LegalHypergraph) -> ImpactAnalysis:
        """
        Analyze the impact of a node on the rest of the graph.
        
        Args:
            node_id: ID of the node to analyze
            graph: Legal hypergraph
            
        Returns:
            ImpactAnalysis: Impact analysis result
        """
        # Check if the node exists
        node = graph.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found in the graph")
        
        # Get direct impacts (nodes directly connected)
        direct_impacts = set()
        for edge_id in node.outgoing_edges:
            edge = graph.get_edge(edge_id)
            if edge:
                direct_impacts.update(edge.target_nodes)
        
        # Get indirect impacts (nodes reachable through paths of length > 1)
        indirect_impacts = self._find_indirect_impacts(node_id, graph, direct_impacts)
        
        # Determine risk level based on impact breadth
        risk_level = self._determine_risk_level(len(direct_impacts), len(indirect_impacts))
        
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
    
    def find_critical_nodes(self, graph: LegalHypergraph) -> List[Dict[str, Any]]:
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
    
    def analyze_relationship_clusters(self, graph: LegalHypergraph) -> List[Dict[str, Any]]:
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
    
    def _reference_matches_clause(self, reference: Dict[str, Any], clause: Dict[str, Any]) -> bool:
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
    
    def _determine_relationship_type(self, 
                                    reference: Dict[str, Any], 
                                    source_clause: Dict[str, Any], 
                                    target_clause: Dict[str, Any]) -> RelationshipType:
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
    
    def _add_definition_relationships(self, 
                                     graph: LegalHypergraph, 
                                     clauses: List[Dict[str, Any]], 
                                     node_id_map: Dict[str, str]) -> None:
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
                import re
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
    
    def _add_conditional_relationships(self, 
                                      graph: LegalHypergraph, 
                                      clauses: List[Dict[str, Any]], 
                                      node_id_map: Dict[str, str]) -> None:
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
    
    def _find_indirect_impacts(self, 
                              node_id: str, 
                              graph: LegalHypergraph, 
                              direct_impacts: Set[str]) -> Set[str]:
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
    
    def _determine_risk_level(self, direct_count: int, indirect_count: int) -> str:
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
    
    def _analyze_cycle_with_llm(self, cycle_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a cycle using the LLM.
        
        Args:
            cycle_data: List of node data dictionaries in the cycle
            
        Returns:
            Dict: Analysis result
        """
        if not self.llm_client:
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
        response = self.llm_client.generate(
            system_prompt="You are a legal document analyzer specializing in identifying circular dependencies and their implications.",
            user_prompt=prompt
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