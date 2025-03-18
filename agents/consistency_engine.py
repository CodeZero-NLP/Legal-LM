from typing import Dict, List, Any, Optional, Tuple
import uuid
from enum import Enum
from dataclasses import dataclass

from utils.ollama_client import OllamaClient
from utils.prompt_templates import PromptTemplates
from utils.response_parser import ResponseParser
from utils.confidence_scorer import ConfidenceScorer
from utils.dependency_analyzer import DependencyAnalyzer
from utils.hypergraph import LegalHypergraph


@dataclass
class Inconsistency:
    """Data class representing an inconsistency between clauses."""
    id: str
    source_clause_id: str
    target_clause_id: str
    description: str
    severity: str
    reasoning: str
    implications: List[str]
    confidence: float


@dataclass
class Dependency:
    """Data class representing a dependency between clauses."""
    source_clause_id: str
    target_clause_id: str
    dependency_type: str
    description: str


@dataclass
class DefinitionIssue:
    """Data class representing an issue with term definitions."""
    term: str
    issue_type: str  # e.g., "undefined", "multiple_definitions", "inconsistent_usage"
    description: str
    affected_clauses: List[str]


class SeverityLevel(Enum):
    """Enum for inconsistency severity levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ContractualConsistencyEngine:
    """
    Engine for checking internal consistency within legal documents.
    Identifies contradictions, definition issues, and logical inconsistencies.
    """
    
    def __init__(self, 
                 llm_client: Any, 
                 min_confidence: float = 0.75,
                 use_hypergraph: bool = True):
        """
        Initialize the ContractualConsistencyEngine.
        
        Args:
            llm_client: Client for LLM interactions
            min_confidence: Minimum confidence threshold for valid issues
            use_hypergraph: Whether to use hypergraph analysis for complex relationships
        """
        self.llm_client = llm_client
        self.prompt_templates = PromptTemplates()
        self.response_parser = ResponseParser()
        self.confidence_scorer = ConfidenceScorer(min_confidence_threshold=min_confidence)
        
        # Initialize dependency analyzer for cross-clause relationships
        self.dependency_analyzer = DependencyAnalyzer()
        
        # Flag for using hypergraph analysis
        self.use_hypergraph = use_hypergraph
    
    def check_consistency(self, 
                         clauses: List[Dict[str, Any]],
                         document_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check consistency across all clauses in a document.
        
        Args:
            clauses: List of clause dictionaries with 'id' and 'text' fields
            document_context: Optional additional context about the document
            
        Returns:
            Dict: Consistency analysis results
        """
        # Generate a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Validate input clauses
        if not clauses or len(clauses) < 2:
            return {
                "analysis_id": analysis_id,
                "inconsistencies": [],
                "inconsistency_count": 0,
                "has_inconsistencies": False,
                "definition_issues": [],
                "definition_issue_count": 0
            }
        
        # Build dependency graph if using hypergraph analysis
        if self.use_hypergraph:
            dependency_graph = self.dependency_analyzer.build_dependency_graph(clauses)
            cycles = dependency_graph.detect_cycles()
            long_range_deps = self.dependency_analyzer.find_long_range_dependencies(dependency_graph)
        else:
            cycles = []
            long_range_deps = []
        
        # Check for definition issues
        definition_issues = self._validate_definitions(clauses)
        
        # Check for pairwise inconsistencies
        inconsistencies = []
        
        # For each clause, check against all other clauses
        for i, source_clause in enumerate(clauses):
            for j, target_clause in enumerate(clauses):
                # Skip self-comparison
                if i == j:
                    continue
                
                # Check consistency between this pair of clauses
                pair_inconsistencies = self._check_clause_pair_consistency(
                    source_clause, 
                    target_clause,
                    document_context
                )
                
                inconsistencies.extend(pair_inconsistencies)
        
        # Add cycle-based inconsistencies
        for cycle in cycles:
            cycle_clauses = [clauses[int(node_id)] for node_id in cycle if node_id.isdigit() and int(node_id) < len(clauses)]
            if cycle_clauses:
                cycle_inconsistency = self._analyze_cycle_inconsistency(cycle_clauses)
                if cycle_inconsistency:
                    inconsistencies.append(cycle_inconsistency)
        
        # Structure the final results
        results = {
            "analysis_id": analysis_id,
            "inconsistencies": inconsistencies,
            "inconsistency_count": len(inconsistencies),
            "has_inconsistencies": len(inconsistencies) > 0,
            "definition_issues": definition_issues,
            "definition_issue_count": len(definition_issues),
            "cycles": cycles if self.use_hypergraph else [],
            "long_range_dependencies": long_range_deps if self.use_hypergraph else []
        }
        
        return results
    
    def _check_clause_pair_consistency(self, 
                                      source_clause: Dict[str, Any], 
                                      target_clause: Dict[str, Any],
                                      document_context: Dict[str, Any] = None) -> List[Inconsistency]:
        """
        Check consistency between a pair of clauses.
        
        Args:
            source_clause: Source clause dictionary
            target_clause: Target clause dictionary
            document_context: Optional document context
            
        Returns:
            List[Inconsistency]: List of detected inconsistencies
        """
        source_id = source_clause.get("id", "unknown")
        source_text = source_clause.get("text", "")
        target_id = target_clause.get("id", "unknown")
        target_text = target_clause.get("text", "")
        
        # Skip if either clause is empty
        if not source_text or not target_text:
            return []
        
        # Format the prompt for consistency checking
        user_prompt = f"""
        Analyze the following pair of clauses for logical inconsistencies, contradictions, or conflicts:
        
        CLAUSE A (ID: {source_id}):
        {source_text}
        
        CLAUSE B (ID: {target_id}):
        {target_text}
        
        Identify any inconsistencies where these clauses contradict each other, create logical conflicts,
        or would be difficult to comply with simultaneously.
        """
        
        # Get consistency analysis from LLM
        response = self.llm_client.generate(
            system_prompt=self.prompt_templates.consistency_check_template,
            user_prompt=user_prompt
        )
        
        # Parse the response to extract issues
        analysis = self.response_parser.parse_consistency_check(response)
        scored_analysis = self.confidence_scorer.score_analysis(analysis)
        
        # Convert issues to Inconsistency objects
        inconsistencies = []
        
        for issue in self.confidence_scorer.get_high_confidence_issues(scored_analysis):
            # Generate a unique ID for this inconsistency
            inconsistency_id = str(uuid.uuid4())
            
            # Determine severity
            severity = issue.get("severity", "MEDIUM")
            
            # Extract implications if available
            implications = []
            reasoning = issue.get("reasoning", "")
            if reasoning:
                implications = self._extract_implications(reasoning)
            
            # Create Inconsistency object
            inconsistency = Inconsistency(
                id=inconsistency_id,
                source_clause_id=source_id,
                target_clause_id=target_id,
                description=issue.get("description", ""),
                severity=severity,
                reasoning=reasoning,
                implications=implications,
                confidence=issue.get("confidence", 0.0)
            )
            
            inconsistencies.append(inconsistency)
        
        return inconsistencies
    
    def _analyze_cycle_inconsistency(self, cycle_clauses: List[Dict[str, Any]]) -> Optional[Inconsistency]:
        """
        Analyze a cycle of clauses for circular dependencies or inconsistencies.
        
        Args:
            cycle_clauses: List of clauses forming a cycle
            
        Returns:
            Optional[Inconsistency]: Inconsistency if found, None otherwise
        """
        # Format the clauses for the prompt
        clauses_text = ""
        clause_ids = []
        
        for i, clause in enumerate(cycle_clauses):
            clause_id = clause.get("id", f"unknown_{i}")
            clause_ids.append(clause_id)
            clauses_text += f"CLAUSE {i+1} (ID: {clause_id}):\n{clause.get('text', '')}\n\n"
        
        # Format the prompt for cycle analysis
        user_prompt = f"""
        Analyze the following cycle of clauses for circular dependencies or logical inconsistencies:
        
        {clauses_text}
        
        These clauses form a dependency cycle. Identify any issues where this circular relationship
        creates logical contradictions, impossible conditions, or implementation challenges.
        """
        
        # Get cycle analysis from LLM
        response = self.llm_client.generate(
            system_prompt="You are a legal document analyzer specializing in detecting circular dependencies and logical inconsistencies.",
            user_prompt=user_prompt
        )
        
        # Check if a significant issue was identified
        if "NO_ISSUES" in response or len(response.strip()) < 50:
            return None
        
        # Create an Inconsistency object for the cycle
        return Inconsistency(
            id=str(uuid.uuid4()),
            source_clause_id=clause_ids[0] if clause_ids else "unknown",
            target_clause_id=clause_ids[-1] if len(clause_ids) > 1 else "unknown",
            description=f"Circular dependency between {len(cycle_clauses)} clauses",
            severity="HIGH",  # Cycles are typically high severity
            reasoning=response,
            implications=self._extract_implications(response),
            confidence=0.9  # High confidence for structural issues
        )
    
    def _validate_definitions(self, clauses: List[Dict[str, Any]]) -> List[DefinitionIssue]:
        """
        Validate term definitions across the document.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            List[DefinitionIssue]: List of definition issues
        """
        # Extract defined terms and their usage
        defined_terms = {}
        term_usage = {}
        
        # First pass: collect defined terms
        for clause in clauses:
            clause_id = clause.get("id", "unknown")
            clause_text = clause.get("text", "")
            
            # Look for definition patterns
            definitions = self._extract_definitions(clause_text)
            
            for term in definitions:
                if term not in defined_terms:
                    defined_terms[term] = []
                defined_terms[term].append({
                    "clause_id": clause_id,
                    "definition": definitions[term]
                })
        
        # Second pass: collect term usage
        for clause in clauses:
            clause_id = clause.get("id", "unknown")
            clause_text = clause.get("text", "")
            
            # Look for term usage
            for term in defined_terms:
                if term in clause_text:
                    if term not in term_usage:
                        term_usage[term] = []
                    term_usage[term].append(clause_id)
        
        # Identify issues
        definition_issues = []
        
        # Check for multiple definitions
        for term, definitions in defined_terms.items():
            if len(definitions) > 1:
                # Check if definitions are consistent
                if not self._are_definitions_consistent(definitions):
                    issue = DefinitionIssue(
                        term=term,
                        issue_type="multiple_definitions",
                        description=f"Term '{term}' has multiple inconsistent definitions",
                        affected_clauses=[d["clause_id"] for d in definitions]
                    )
                    definition_issues.append(issue)
        
        # Check for undefined but used terms
        # This would require a more sophisticated analysis of legal terminology
        # For now, we'll focus on inconsistent definitions
        
        return definition_issues
    
    def _extract_definitions(self, text: str) -> Dict[str, str]:
        """
        Extract defined terms from clause text.
        
        Args:
            text: Clause text
            
        Returns:
            Dict[str, str]: Dictionary of terms and their definitions
        """
        # Use the LLM to extract defined terms
        prompt = f"""
        Extract all defined terms from the following clause text.
        A defined term is typically indicated by quotes, capitalization, or explicit definition.
        
        CLAUSE TEXT:
        {text}
        
        Format your response as:
        [TERM]
        Term: <term>
        Definition: <definition>
        [/TERM]
        
        If no terms are defined, respond with "NO_DEFINED_TERMS".
        """
        
        response = self.llm_client.generate(
            system_prompt="You are a legal document analyzer specializing in extracting defined terms.",
            user_prompt=prompt
        )
        
        # Parse the response
        definitions = {}
        
        if "NO_DEFINED_TERMS" in response:
            return definitions
        
        term_blocks = response.split("[TERM]")
        for block in term_blocks:
            if "[/TERM]" not in block:
                continue
                
            content = block.split("[/TERM]")[0].strip()
            term = ""
            definition = ""
            
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("Term:"):
                    term = line[5:].strip()
                elif line.startswith("Definition:"):
                    definition = line[11:].strip()
            
            if term and definition:
                definitions[term] = definition
        
        return definitions
    
    def _are_definitions_consistent(self, definitions: List[Dict[str, Any]]) -> bool:
        """
        Check if multiple definitions of a term are consistent.
        
        Args:
            definitions: List of definition dictionaries
            
        Returns:
            bool: True if definitions are consistent, False otherwise
        """
        if len(definitions) <= 1:
            return True
        
        # Extract the definition texts
        definition_texts = [d["definition"] for d in definitions]
        
        # Use the LLM to check consistency
        prompt = f"""
        Determine if the following definitions of the same term are consistent with each other:
        
        {"".join([f"Definition {i+1}: {text}\n" for i, text in enumerate(definition_texts)])}
        
        Respond with "CONSISTENT" if the definitions are compatible and don't contradict each other.
        Respond with "INCONSISTENT" if the definitions contradict or are incompatible with each other.
        """
        
        response = self.llm_client.generate(
            system_prompt="You are a legal document analyzer specializing in term definitions.",
            user_prompt=prompt
        )
        
        return "CONSISTENT" in response.upper()
    
    def _extract_implications(self, text: str) -> List[str]:
        """
        Extract implications from analysis text.
        
        Args:
            text: Analysis text
            
        Returns:
            List[str]: Extracted implications
        """
        # Look for implication patterns in the text
        implications = []
        
        # Split into sentences
        sentences = text.split(". ")
        
        # Look for implication indicators
        indicators = ["could lead to", "may result in", "implies", "means that", 
                     "consequence", "impact", "effect", "implication"]
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in indicators):
                implications.append(sentence.strip())
        
        # If no implications found using indicators, use the LLM
        if not implications and len(text) > 50:
            prompt = f"""
            Extract the key implications from the following analysis:
            
            {text}
            
            List each implication on a separate line starting with "- ".
            If no clear implications are present, respond with "NO_IMPLICATIONS".
            """
            
            response = self.llm_client.generate(
                system_prompt="You are a legal analyst specializing in identifying implications.",
                user_prompt=prompt
            )
            
            if "NO_IMPLICATIONS" not in response:
                for line in response.split("\n"):
                    line = line.strip()
                    if line.startswith("- "):
                        implications.append(line[2:])
        
        return implications
    
    def analyze_dependencies(self, 
                            clause: Dict[str, Any], 
                            all_clauses: List[Dict[str, Any]]) -> List[Dependency]:
        """
        Analyze dependencies between a clause and all other clauses.
        
        Args:
            clause: The clause to analyze
            all_clauses: All clauses in the document
            
        Returns:
            List[Dependency]: List of dependencies
        """
        clause_id = clause.get("id", "unknown")
        clause_text = clause.get("text", "")
        
        # Extract references from the clause text
        references = self.dependency_analyzer.extract_references(clause_text)
        
        # Map references to target clauses
        dependencies = []
        
        for reference in references:
            # Try to find matching clauses
            for target_clause in all_clauses:
                target_id = target_clause.get("id", "unknown")
                target_text = target_clause.get("text", "")
                
                # Skip self-references
                if target_id == clause_id:
                    continue
                
                # Check if the reference matches this target clause
                if self._reference_matches_clause(reference, target_clause):
                    dependency = Dependency(
                        source_clause_id=clause_id,
                        target_clause_id=target_id,
                        dependency_type=reference["type"],
                        description=f"Reference '{reference['full_text']}' in clause {clause_id} points to clause {target_id}"
                    )
                    dependencies.append(dependency)
        
        return dependencies
    
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