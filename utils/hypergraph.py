from typing import Dict, List, Any, Set, Tuple, Optional
import uuid

class HyperNode:
    """
    Node in a legal hypergraph representing a legal entity (clause, definition, etc.)
    """
    
    def __init__(self, id: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """
        Initialize a hypernode.
        
        Args:
            id: Unique identifier for the node
            data: Data associated with the node
        """
        self.id = id or str(uuid.uuid4())
        self.data = data or {}
        self.incoming_edges = set()
        self.outgoing_edges = set()
    
    def add_data(self, key: str, value: Any) -> None:
        """Add data to the node."""
        self.data[key] = value
    
    def add_incoming_edge(self, edge_id: str) -> None:
        """Add an incoming edge ID to the node."""
        self.incoming_edges.add(edge_id)
    
    def add_outgoing_edge(self, edge_id: str) -> None:
        """Add an outgoing edge ID to the node."""
        self.outgoing_edges.add(edge_id)

class HyperEdge:
    """
    Edge in a legal hypergraph representing relationships between legal entities.
    Can connect multiple source nodes to multiple target nodes.
    """
    
    def __init__(self, id: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """
        Initialize a hyperedge.
        
        Args:
            id: Unique identifier for the edge
            data: Data associated with the edge
        """
        self.id = id or str(uuid.uuid4())
        self.data = data or {}
        self.source_nodes = set()
        self.target_nodes = set()
    
    def add_data(self, key: str, value: Any) -> None:
        """Add data to the edge."""
        self.data[key] = value
    
    def add_source(self, node_id: str) -> None:
        """Add a source node ID to the edge."""
        self.source_nodes.add(node_id)
    
    def add_target(self, node_id: str) -> None:
        """Add a target node ID to the edge."""
        self.target_nodes.add(node_id)

class LegalHypergraph:
    """
    Hypergraph implementation for modeling complex legal relationships.
    """
    
    def __init__(self):
        """Initialize an empty legal hypergraph."""
        self.nodes: Dict[str, HyperNode] = {}
        self.edges: Dict[str, HyperEdge] = {}
    
    def add_node(self, node_data: Dict[str, Any], node_id: Optional[str] = None) -> str:
        """
        Add a node to the hypergraph.
        
        Args:
            node_data: Data to associate with the node
            node_id: Optional custom node ID
            
        Returns:
            str: ID of the created node
        """
        node = HyperNode(id=node_id, data=node_data)
        self.nodes[node.id] = node
        return node.id
    
    def add_edge(self, 
                edge_type: str, 
                source_nodes: List[str], 
                target_nodes: List[str], 
                edge_data: Optional[Dict[str, Any]] = None,
                edge_id: Optional[str] = None) -> str:
        """
        Add an edge to the hypergraph.
        
        Args:
            edge_type: Type of relationship
            source_nodes: List of source node IDs
            target_nodes: List of target node IDs
            edge_data: Optional data to associate with the edge
            edge_id: Optional custom edge ID
            
        Returns:
            str: ID of the created edge
        """
        if edge_data is None:
            edge_data = {}
        edge_data["type"] = edge_type
        
        edge = HyperEdge(id=edge_id, data=edge_data)
        
        # Connect source nodes to edge
        for node_id in source_nodes:
            if node_id in self.nodes:
                edge.add_source(node_id)
                self.nodes[node_id].add_outgoing_edge(edge.id)
            else:
                raise ValueError(f"Source node {node_id} does not exist in the graph")
        
        # Connect edge to target nodes
        for node_id in target_nodes:
            if node_id in self.nodes:
                edge.add_target(node_id)
                self.nodes[node_id].add_incoming_edge(edge.id)
            else:
                raise ValueError(f"Target node {node_id} does not exist in the graph")
        
        self.edges[edge.id] = edge
        return edge.id
    
    def get_node(self, node_id: str) -> Optional[HyperNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[HyperEdge]:
        """Get an edge by ID."""
        return self.edges.get(edge_id)
    
    def get_connected_nodes(self, node_id: str) -> Dict[str, Set[str]]:
        """
        Get all nodes connected to the specified node.
        
        Args:
            node_id: ID of the node to get connections for
            
        Returns:
            Dict: Dictionary with 'incoming' and 'outgoing' node sets
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph")
        
        node = self.nodes[node_id]
        
        incoming_nodes = set()
        for edge_id in node.incoming_edges:
            edge = self.edges.get(edge_id)
            if edge:
                incoming_nodes.update(edge.source_nodes)
        
        outgoing_nodes = set()
        for edge_id in node.outgoing_edges:
            edge = self.edges.get(edge_id)
            if edge:
                outgoing_nodes.update(edge.target_nodes)
        
        return {
            'incoming': incoming_nodes,
            'outgoing': outgoing_nodes
        }
    
    def detect_cycles(self) -> List[List[str]]:
        """
        Detect cycles in the hypergraph.
        
        Returns:
            List[List[str]]: List of cycles, where each cycle is a list of node IDs
        """
        cycles = []
        visited = set()
        path = []
        path_set = set()
        
        def dfs(node_id: str) -> None:
            if node_id in path_set:
                # Found a cycle
                cycle_start = path.index(node_id)
                cycles.append(path[cycle_start:] + [node_id])
                return
            
            if node_id in visited:
                return
            
            visited.add(node_id)
            path.append(node_id)
            path_set.add(node_id)
            
            node = self.nodes.get(node_id)
            if node:
                for edge_id in node.outgoing_edges:
                    edge = self.edges.get(edge_id)
                    if edge:
                        for target_id in edge.target_nodes:
                            dfs(target_id)
            
            path.pop()
            path_set.remove(node_id)
        
        # Start DFS from each node
        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id)
        
        return cycles 