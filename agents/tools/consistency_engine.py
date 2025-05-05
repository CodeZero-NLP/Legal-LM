import time
from typing import Dict, List, Any, Optional, Tuple
import uuid
from enum import Enum
from dataclasses import dataclass

from agents.utils.ollama_client import OllamaClient
from agents.utils.prompt_templates import PromptTemplates
from agents.utils.response_parser import ResponseParser
from agents.utils.confidence_scorer import ConfidenceScorer
from agents.utils.dependency_analyzer import DependencyAnalyzer


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


def check_contractual_consistency(
    clauses: List[Dict[str, Any]],
    llm_client: Any, 
    document_context: Dict[str, Any] = None,
    min_confidence: float = 0.75,
    use_hypergraph: bool = True
) -> Dict[str, Any]:
    """
    Check consistency across all clauses in a document.
    
    This function:
    1. Analyzes term definitions across the document
    2. Performs pairwise consistency checks between clauses
    3. Detects circular dependencies if hypergraph analysis is enabled
    4. Summarizes all inconsistencies and definition issues
    
    Args:
        clauses: List of clause dictionaries with 'id' and 'text' fields
        llm_client: Client for LLM interactions
        document_context: Optional additional context about the document
        min_confidence: Minimum confidence threshold for valid issues (default: 0.75)
        use_hypergraph: Whether to use hypergraph analysis for complex relationships (default: True)
        
    Returns:
        Dict: Consistency analysis results including inconsistencies, definition issues, etc.
    """
    # Initialize utility components
    prompt_templates = PromptTemplates()
    response_parser = ResponseParser()
    confidence_scorer = ConfidenceScorer(min_confidence_threshold=min_confidence)
    dependency_analyzer = DependencyAnalyzer()
    
    # Generate a unique ID for this analysis
    analysis_id = str(uuid.uuid4())
    
    print(f"Starting contractual consistency check for {len(clauses)} clauses (min_confidence={min_confidence})")
    
    # Helper function: Extract implications from text
    def extract_implications(text: str) -> List[str]:
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
            
            try:
                # Adding a delay here to avoid hitting API rate limits
                time.sleep(5)
                
                response = llm_client.query(
                    system_prompt="You are a legal analyst specializing in identifying implications.",
                    prompt=prompt
                )
                
                if "NO_IMPLICATIONS" not in response:
                    for line in response.split("\n"):
                        line = line.strip()
                        if line.startswith("- "):
                            implications.append(line[2:])
            except Exception as e:
                print(f"ERROR: Failed to extract implications: {e}")
        
        return implications
    
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
    
    # Helper function: Analyze dependencies
    def analyze_dependencies(clause: Dict[str, Any], all_clauses: List[Dict[str, Any]]) -> List[Dependency]:
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
        references = dependency_analyzer.extract_references(clause_text)
        
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
                if reference_matches_clause(reference, target_clause):
                    dependency = Dependency(
                        source_clause_id=clause_id,
                        target_clause_id=target_id,
                        dependency_type=reference["type"],
                        description=f"Reference '{reference['full_text']}' in clause {clause_id} points to clause {target_id}"
                    )
                    dependencies.append(dependency)
        
        return dependencies
    
    # Helper function: Extract definitions
    def extract_definitions(text: str) -> Dict[str, str]:
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
        
        try:
            # Adding a delay here to avoid hitting API rate limits
            time.sleep(5)

            response = llm_client.query(
                system_prompt="You are a legal document analyzer specializing in extracting defined terms.",
                prompt=prompt
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
        except Exception as e:
            print(f"ERROR: Failed to extract definitions: {e}")
            return {}
    
    # Helper function: Check if definitions are consistent
    def are_definitions_consistent(definitions: List[Dict[str, Any]]) -> bool:
        """
        Check if multiple definitions of the same term are consistent.
        
        Args:
            definitions: List of definition dictionaries
            
        Returns:
            bool: True if the definitions are consistent
        """
        # Extract the definition texts
        definition_texts = [d["definition"] for d in definitions]
        
        # Use the LLM to check consistency
        definitions_text = ""
        for i, text in enumerate(definition_texts):
            definitions_text += f"Definition {i+1}: {text}\n"
        
        prompt = f"""
        Determine if the following definitions of the same term are consistent with each other:
        
        {definitions_text}
        
        Respond with "CONSISTENT" if the definitions are compatible and don't contradict each other.
        Respond with "INCONSISTENT" if the definitions contradict or are incompatible with each other.
        """
        
        try:
            # Adding a delay here to avoid hitting API rate limits
            time.sleep(5)

            response = llm_client.query(
                system_prompt="You are a legal document analyzer specializing in term definitions.",
                prompt=prompt
            )
            
            return "CONSISTENT" in response.upper()
        except Exception as e:
            print(f"ERROR: Failed to check definition consistency: {e}")
            return True  # Default to consistent if check fails
    
    # Helper function: Validate term definitions
    def validate_definitions(clauses: List[Dict[str, Any]]) -> List[DefinitionIssue]:
        """
        Validate term definitions across the document.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            List[DefinitionIssue]: List of definition issues
        """
        print("Validating term definitions across document")
        
        # Extract defined terms and their usage
        defined_terms = {}
        term_usage = {}
        
        # First pass: collect defined terms
        for clause in clauses:
            clause_id = clause.get("id", "unknown")
            clause_text = clause.get("text", "")
            
            # Look for definition patterns
            definitions = extract_definitions(clause_text)
            
            for term in definitions:
                if term not in defined_terms:
                    defined_terms[term] = []
                defined_terms[term].append({
                    "clause_id": clause_id,
                    "definition": definitions[term]
                })
        
        print(f"Found {len(defined_terms)} defined terms across the document")
        
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
                if not are_definitions_consistent(definitions):
                    issue = DefinitionIssue(
                        term=term,
                        issue_type="multiple_definitions",
                        description=f"Term '{term}' has multiple inconsistent definitions",
                        affected_clauses=[d["clause_id"] for d in definitions]
                    )
                    definition_issues.append(issue)
                    print(f"Found inconsistent definitions for term '{term}'")
        
        print(f"Identified {len(definition_issues)} term definition issues")
        return definition_issues
    
    # Helper function: Check consistency between clause pairs
    def check_clause_pair_consistency(
        source_clause: Dict[str, Any], 
        target_clause: Dict[str, Any],
        document_context: Dict[str, Any] = None
    ) -> List[Inconsistency]:
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
        try:
            # Adding a delay here to avoid hitting API rate limits
            time.sleep(5)

            response = llm_client.query(
                system_prompt=prompt_templates.consistency_check_template,
                prompt=user_prompt
            )
            
            # Parse the response to extract issues
            analysis = response_parser.parse_consistency_check(response)
            scored_analysis = confidence_scorer.score_analysis(analysis)
            
            # Convert issues to Inconsistency objects
            inconsistencies = []
            
            for issue in confidence_scorer.get_high_confidence_issues(scored_analysis):
                # Generate a unique ID for this inconsistency
                inconsistency_id = str(uuid.uuid4())
                
                # Determine severity
                severity = issue.get("severity", "MEDIUM")
                
                # Extract implications if available
                implications = []
                reasoning = issue.get("reasoning", "")
                if reasoning:
                    implications = extract_implications(reasoning)
                
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
            
            if inconsistencies:
                print(f"Found {len(inconsistencies)} inconsistencies between clauses {source_id} and {target_id}")
            
            return inconsistencies
        except Exception as e:
            print(f"ERROR: Failed to check clause pair consistency: {e}")
            return []
    
    # Helper function: Analyze a cycle of clauses
    def analyze_cycle_inconsistency(cycle_clauses: List[Dict[str, Any]]) -> Optional[Inconsistency]:
        """
        Analyze a cycle of clauses for circular dependencies or inconsistencies.
        
        Args:
            cycle_clauses: List of clauses forming a cycle
            
        Returns:
            Optional[Inconsistency]: Inconsistency if found, None otherwise
        """
        print(f"Analyzing cycle of {len(cycle_clauses)} clauses for circular dependencies")
        
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
        try:
            # Adding a delay here to avoid hitting API rate limits
            time.sleep(5)

            response = llm_client.query(
                system_prompt="You are a legal document analyzer specializing in detecting circular dependencies and logical inconsistencies.",
                prompt=user_prompt
            )
            
            # Check if a significant issue was identified
            if "NO_ISSUES" in response or len(response.strip()) < 50:
                print("No significant issues found in the cycle")
                return None
            
            # Create an Inconsistency object for the cycle
            inconsistency = Inconsistency(
                id=str(uuid.uuid4()),
                source_clause_id=clause_ids[0] if clause_ids else "unknown",
                target_clause_id=clause_ids[-1] if len(clause_ids) > 1 else "unknown",
                description=f"Circular dependency between {len(cycle_clauses)} clauses",
                severity="HIGH",  # Cycles are typically high severity
                reasoning=response,
                implications=extract_implications(response),
                confidence=0.9  # High confidence for structural issues
            )
            
            print(f"Found circular dependency issue with severity HIGH")
            return inconsistency
        except Exception as e:
            print(f"ERROR: Failed to analyze cycle inconsistency: {e}")
            return None
    
    # MAIN FUNCTION EXECUTION
    
    # Validate input clauses
    if not clauses or len(clauses) < 2:
        print("WARNING: Not enough clauses for consistency analysis")
        return {
            "analysis_id": analysis_id,
            "inconsistencies": [],
            "inconsistency_count": 0,
            "has_inconsistencies": False,
            "definition_issues": [],
            "definition_issue_count": 0
        }
    
    # Build dependency graph if using hypergraph analysis
    if use_hypergraph:
        print("Building dependency graph for hypergraph analysis")
        dependency_graph = dependency_analyzer.build_dependency_graph(clauses)
        cycles = dependency_graph.detect_cycles()
        if cycles:
            print(f"Detected {len(cycles)} cycles in the dependency graph")
        long_range_deps = dependency_analyzer.find_long_range_dependencies(dependency_graph)
        if long_range_deps:
            print(f"Found {len(long_range_deps)} long-range dependencies")
    else:
        print("Skipping hypergraph analysis")
        cycles = []
        long_range_deps = []
    
    # Check for definition issues
    definition_issues = validate_definitions(clauses)
    
    # Check for pairwise inconsistencies
    inconsistencies = []
    
    print("Checking pairwise consistency between clauses")
    # For each clause, check against all other clauses
    for i, source_clause in enumerate(clauses):
        for j, target_clause in enumerate(clauses):
            # Skip self-comparison
            if i == j:
                continue
            
            # Check consistency between this pair of clauses
            pair_inconsistencies = check_clause_pair_consistency(
                source_clause, 
                target_clause,
                document_context
            )
            
            inconsistencies.extend(pair_inconsistencies)
    
    # Add cycle-based inconsistencies
    if cycles:
        print("Analyzing detected cycles for inconsistencies")
        for cycle in cycles:
            cycle_clauses = [clauses[int(node_id)] for node_id in cycle if node_id.isdigit() and int(node_id) < len(clauses)]
            if cycle_clauses:
                cycle_inconsistency = analyze_cycle_inconsistency(cycle_clauses)
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
        "cycles": cycles if use_hypergraph else [],
        "long_range_dependencies": long_range_deps if use_hypergraph else []
    }
    
    print(f"Completed consistency analysis: {len(inconsistencies)} inconsistencies, {len(definition_issues)} definition issues")
    return results

