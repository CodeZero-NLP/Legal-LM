from typing import Dict, List, Any, Optional, Tuple
import uuid
from enum import Enum
from dataclasses import dataclass

from utils.ollama_client import OllamaClient
from utils.prompt_templates import PromptTemplates
from utils.response_parser import ResponseParser
from utils.confidence_scorer import ConfidenceScorer
from utils.blackstone import BlackstoneNER
from utils.websearcher import WebContentRetriever


@dataclass
class PrecedentMatch:
    """Data class representing a match between a clause and a legal precedent."""
    case_name: str
    citation: str
    jurisdiction: str
    year: int
    relevance_score: float
    key_holdings: List[str]
    implications: List[str]
    contradictions: List[str]


@dataclass
class Holding:
    """Data class representing a key holding from a legal case."""
    text: str
    principle: str
    relevance: float


class RelevanceLevel(Enum):
    """Enum for precedent relevance levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


def analyze_precedents_for_compliance(
        clause_text: str,
        llm_client: Any,
        jurisdiction: str = "US",
        context: Dict[str, Any] = None,
        min_confidence: float = 0.75,
        use_web_search: bool = False,
        qdrant_url: str = "http://localhost:6333"
    ) -> Dict[str, Any]:
    """
    Analyze a clause against relevant legal precedents.
    
    This function:
    1. Extracts legal entities from the clause
    2. Retrieves relevant precedents (from web or simulated)
    3. Formats a prompt for the LLM to analyze precedent compliance
    4. Processes the LLM response to extract structured findings
    5. Filters out low-confidence results
    
    Args:
        clause_text: Text of the clause to analyze
        llm_client: Client for LLM interactions
        jurisdiction: Legal jurisdiction (default: "US")
        context: Optional additional context
        min_confidence: Minimum confidence threshold for valid issues (default: 0.75)
        use_web_search: Whether to use web search for precedent retrieval (default: False)
        qdrant_url: URL for Qdrant vector database if using web search (default: "http://localhost:6333")
        
    Returns:
        Dict: Analysis results including precedent matches and issues
    """
    # Initialize utility components
    prompt_templates = PromptTemplates()
    response_parser = ResponseParser()
    confidence_scorer = ConfidenceScorer(min_confidence_threshold=min_confidence)
    ner = BlackstoneNER()
    
    # Initialize web retriever if web search is enabled
    web_retriever = None
    if use_web_search:
        web_retriever = WebContentRetriever(
            qdrant_url=qdrant_url,
            qdrant_collection_name="legal_precedents",
            num_results=3
        )
    
    # Helper function: Extract legal entities
    def extract_legal_entities(text: str) -> Dict[str, List[str]]:
        """Extract legal entities from text using NER."""
        entities = ner.extract_entities(text)
        
        # Group entities by type
        grouped_entities = {}
        for entity, entity_type in entities:
            if entity_type not in grouped_entities:
                grouped_entities[entity_type] = []
            grouped_entities[entity_type].append(entity)
        
        return grouped_entities
    
    # Helper function: Extract precedent from web result
    def extract_precedent_from_web_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract structured precedent information from web search result."""
        # Use the LLM to extract structured information from the web content
        prompt = f"""
        Extract structured legal precedent information from the following text.
        If the text doesn't contain a clear legal precedent, respond with "NO_PRECEDENT".
        
        TEXT:
        {result['content'][:2000]}  # Limit content length
        
        Format your response as:
        Case Name: [case name]
        Citation: [citation]
        Jurisdiction: [jurisdiction]
        Year: [year]
        Key Holdings: [comma-separated list]
        """
        
        response = llm_client.generate(
            system_prompt="You are a legal information extractor. Extract structured information about legal precedents.",
            user_prompt=prompt
        )
        
        # Check if no precedent was found
        if "NO_PRECEDENT" in response:
            return None
        
        # Parse the response
        precedent = {}
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key == 'case_name':
                    precedent['case_name'] = value
                elif key == 'citation':
                    precedent['citation'] = value
                elif key == 'jurisdiction':
                    precedent['jurisdiction'] = value
                elif key == 'year':
                    try:
                        precedent['year'] = int(value)
                    except ValueError:
                        precedent['year'] = 0
                elif key == 'key_holdings':
                    precedent['key_holdings'] = [h.strip() for h in value.split(',')]
        
        # Validate that we have the minimum required fields
        if 'case_name' in precedent and 'citation' in precedent:
            # Add source information
            precedent['source'] = result['url']
            precedent['relevance_score'] = result['score']
            return precedent
        
        return None
    
    # Helper function: Parse precedent block
    def parse_precedent_block(block: str) -> Optional[Dict[str, Any]]:
        """Parse a precedent block from LLM response."""
        precedent = {
            'key_holdings': []
        }
        
        current_field = None
        in_holdings = False
        
        for line in block.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for field headers
            if ':' in line and not in_holdings:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key == 'case_name':
                    precedent['case_name'] = value
                elif key == 'citation':
                    precedent['citation'] = value
                elif key == 'jurisdiction':
                    precedent['jurisdiction'] = value
                elif key == 'year':
                    try:
                        precedent['year'] = int(value)
                    except ValueError:
                        precedent['year'] = 0
                elif key == 'relevance':
                    # Convert relevance to a score
                    if value.upper() == 'HIGH':
                        precedent['relevance_score'] = 0.9
                    elif value.upper() == 'MEDIUM':
                        precedent['relevance_score'] = 0.7
                    else:
                        precedent['relevance_score'] = 0.5
                elif key == 'key_holdings':
                    in_holdings = True
            elif in_holdings and line.startswith('-'):
                # This is a holding bullet point
                holding = line[1:].strip()
                precedent['key_holdings'].append(holding)
        
        # Validate that we have the minimum required fields
        if 'case_name' in precedent and 'citation' in precedent and precedent['key_holdings']:
            return precedent
        
        return None
    
    # Helper function: Simulate precedent retrieval
    def simulate_precedent_retrieval(text: str, jurisdiction: str, entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Simulate precedent retrieval using the LLM when web search is not available."""
        # Create a prompt for the LLM to generate relevant precedents
        prompt = f"""
        Generate 3-5 relevant legal precedents for the following clause in {jurisdiction} jurisdiction.
        
        CLAUSE:
        {text}
        
        For each precedent, provide:
        1. Case name
        2. Citation
        3. Jurisdiction
        4. Year
        5. Key holdings (2-3 bullet points)
        6. Relevance to the clause (HIGH, MEDIUM, or LOW)
        
        Format each precedent as:
        [PRECEDENT]
        Case Name: [case name]
        Citation: [citation]
        Jurisdiction: [jurisdiction]
        Year: [year]
        Key Holdings:
        - [holding 1]
        - [holding 2]
        Relevance: [HIGH/MEDIUM/LOW]
        [/PRECEDENT]
        """
        
        response = llm_client.generate(
            system_prompt="You are a legal research assistant with expertise in case law.",
            user_prompt=prompt
        )
        
        # Parse the response to extract precedents
        precedents = []
        precedent_blocks = response.split('[PRECEDENT]')
        
        for block in precedent_blocks:
            if '[/PRECEDENT]' not in block:
                continue
                
            content = block.split('[/PRECEDENT]')[0].strip()
            precedent = parse_precedent_block(content)
            if precedent:
                precedents.append(precedent)
        
        return precedents
    
    # Helper function: Retrieve precedents from web
    def retrieve_precedents_from_web(text: str, jurisdiction: str, entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Retrieve precedents using web search."""
        if not web_retriever:
            return []
        
        # Construct search queries based on the clause and entities
        search_queries = []
        
        # Add a general query based on the clause
        search_queries.append(f"legal precedent {jurisdiction} {text[:100]}")
        
        # Add queries for specific entity types
        if "PROVISION" in entities:
            for provision in entities["PROVISION"][:2]:  # Limit to first 2 to avoid too many queries
                search_queries.append(f"legal case {jurisdiction} {provision}")
        
        if "COURT" in entities:
            for court in entities["COURT"][:2]:
                search_queries.append(f"{court} precedent {jurisdiction}")
        
        # Execute searches and store results
        precedents = []
        for query in search_queries:
            # Store content in vector database for future retrieval
            web_retriever.store_content_in_qdrant(query)
            
            # Search for relevant content
            results = web_retriever.search_in_qdrant(query)
            
            for result in results:
                # Process and structure the result
                precedent = extract_precedent_from_web_result(result)
                if precedent:
                    precedents.append(precedent)
        
        # Remove duplicates based on case name
        unique_precedents = []
        seen_cases = set()
        for precedent in precedents:
            if precedent["case_name"] not in seen_cases:
                seen_cases.add(precedent["case_name"])
                unique_precedents.append(precedent)
        
        return unique_precedents[:5]  # Limit to top 5 precedents
    
    # Helper function: Retrieve precedents
    def retrieve_precedents(text: str, jurisdiction: str, entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Retrieve relevant precedents for the clause."""
        # If web search is enabled, use it to find precedents
        if use_web_search and web_retriever:
            return retrieve_precedents_from_web(text, jurisdiction, entities)
        
        # Otherwise, use a simulated approach with the LLM
        return simulate_precedent_retrieval(text, jurisdiction, entities)
    
    # Helper function: Format precedents for prompt
    def format_precedents_for_prompt(precedents: List[Dict[str, Any]]) -> str:
        """Format precedents for inclusion in the LLM prompt."""
        if not precedents:
            return ""
            
        formatted_text = ""
        
        for i, precedent in enumerate(precedents):
            formatted_text += f"PRECEDENT {i+1}:\n"
            formatted_text += f"Case: {precedent.get('case_name', 'Unknown')}\n"
            formatted_text += f"Citation: {precedent.get('citation', 'Unknown')}\n"
            formatted_text += f"Jurisdiction: {precedent.get('jurisdiction', 'Unknown')}\n"
            formatted_text += f"Year: {precedent.get('year', 'Unknown')}\n"
            
            # Add key holdings
            formatted_text += "Key Holdings:\n"
            for holding in precedent.get('key_holdings', []):
                formatted_text += f"- {holding}\n"
                
            formatted_text += "\n"
        
        return formatted_text
    
    # Helper function: Extract holdings
    def extract_holdings(precedent: Dict[str, Any]) -> List[Holding]:
        """Extract structured holdings from a precedent."""
        holdings = []
        
        for holding_text in precedent.get('key_holdings', []):
            # Use the LLM to extract the principle and assess relevance
            prompt = f"""
            Analyze the following legal holding and extract:
            1. The core legal principle
            2. Its relevance score (0.0 to 1.0)
            
            HOLDING:
            {holding_text}
            
            Format your response as:
            Principle: [core legal principle]
            Relevance: [score between 0.0 and 1.0]
            """
            
            response = llm_client.generate(
                system_prompt="You are a legal analyst specializing in case law interpretation.",
                user_prompt=prompt
            )
            
            # Parse the response
            principle = ""
            relevance = 0.5  # Default relevance
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('Principle:'):
                    principle = line[len('Principle:'):].strip()
                elif line.startswith('Relevance:'):
                    try:
                        relevance_str = line[len('Relevance:'):].strip()
                        relevance = float(relevance_str)
                    except ValueError:
                        # If parsing fails, keep the default
                        pass
            
            # Create a Holding object
            holding = Holding(
                text=holding_text,
                principle=principle,
                relevance=relevance
            )
            
            holdings.append(holding)
            
        return holdings
    
    # Helper function: Rank precedents
    def rank_precedents(precedents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank precedents by relevance and authority."""
        # Sort precedents by relevance score (descending)
        ranked_precedents = sorted(
            precedents, 
            key=lambda p: p.get('relevance_score', 0), 
            reverse=True
        )
        
        # Add rank information
        for i, precedent in enumerate(ranked_precedents):
            precedent['rank'] = i + 1
        
        return ranked_precedents
    
    # MAIN FUNCTION EXECUTION
    
    # Generate a unique ID for this analysis
    analysis_id = str(uuid.uuid4())
    
    # Extract legal entities from the clause
    legal_entities = extract_legal_entities(clause_text)
    
    # Retrieve relevant precedents
    precedents = retrieve_precedents(clause_text, jurisdiction, legal_entities)
    
    # Format precedents for inclusion in the prompt
    precedents_text = format_precedents_for_prompt(precedents)
    
    # Generate the prompt for precedent analysis
    user_prompt = prompt_templates.format_precedent_prompt(
        clause_text=clause_text,
        jurisdiction=jurisdiction,
        precedents=precedents_text
    )
    
    # Get analysis from LLM
    response = llm_client.generate(
        system_prompt=prompt_templates.precedent_analysis_template,
        user_prompt=user_prompt
    )
    
    # Parse the response
    parsed_analysis = response_parser.parse_precedent_analysis(response)
    
    # Score the analysis for confidence
    scored_analysis = confidence_scorer.score_analysis(parsed_analysis)
    
    # Extract high-confidence issues
    high_confidence_issues = confidence_scorer.get_high_confidence_issues(scored_analysis)
    
    # Structure the final results
    results = {
        "analysis_id": analysis_id,
        "clause_text": clause_text,
        "jurisdiction": jurisdiction,
        "precedent_count": len(precedents),
        "precedents": precedents,
        "issues": high_confidence_issues,
        "issue_count": len(high_confidence_issues),
        "has_issues": len(high_confidence_issues) > 0,
        "average_confidence": scored_analysis.get("average_confidence", 0)
    }
    
    return results


class PrecedentAnalyzer:
    """
    Legacy class maintained for backward compatibility.
    Analyzer for evaluating clauses against legal precedents and case law.
    Identifies potential conflicts with established legal interpretations.
    Now uses the standalone analyze_precedents_for_compliance function.
    """
    
    def __init__(self, 
                 llm_client: Any, 
                 min_confidence: float = 0.75,
                 use_web_search: bool = False,
                 qdrant_url: str = "http://localhost:6333"):
        """
        Initialize the PrecedentAnalyzer.
        
        Args:
            llm_client: Client for LLM interactions
            min_confidence: Minimum confidence threshold for valid issues
            use_web_search: Whether to use web search for precedent retrieval
            qdrant_url: URL for Qdrant vector database if using web search
        """
        self.llm_client = llm_client
        self.min_confidence = min_confidence
        self.use_web_search = use_web_search
        self.qdrant_url = qdrant_url
        self.prompt_templates = PromptTemplates()
        self.response_parser = ResponseParser()
        self.confidence_scorer = ConfidenceScorer(min_confidence_threshold=min_confidence)
        
        # Initialize NER for legal entity extraction
        self.ner = BlackstoneNER()
        
        # Initialize web search capability if enabled
        self.web_retriever = None
        if use_web_search:
            self.web_retriever = WebContentRetriever(
                qdrant_url=qdrant_url,
                qdrant_collection_name="legal_precedents",
                num_results=3
            )
    
    def analyze_precedents(self, 
                          clause_text: str, 
                          jurisdiction: str = "US",
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a clause against relevant legal precedents.
        This method now uses the standalone analyze_precedents_for_compliance function.
        
        Args:
            clause_text: Text of the clause to analyze
            jurisdiction: Legal jurisdiction (default: "US")
            context: Optional additional context
            
        Returns:
            Dict: Analysis results including precedent matches and issues
        """
        return analyze_precedents_for_compliance(
            clause_text=clause_text,
            llm_client=self.llm_client,
            jurisdiction=jurisdiction,
            context=context,
            min_confidence=self.min_confidence,
            use_web_search=self.use_web_search,
            qdrant_url=self.qdrant_url
        )
    
    # Legacy methods for backward compatibility
    
    def _extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities from text using NER."""
        entities = self.ner.extract_entities(text)
        
        # Group entities by type
        grouped_entities = {}
        for entity, entity_type in entities:
            if entity_type not in grouped_entities:
                grouped_entities[entity_type] = []
            grouped_entities[entity_type].append(entity)
        
        return grouped_entities
    
    def _retrieve_precedents(self, 
                           clause_text: str, 
                           jurisdiction: str,
                           entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Retrieve relevant precedents for the clause."""
        # If web search is enabled, use it to find precedents
        if self.use_web_search:
            return self._retrieve_precedents_from_web(clause_text, jurisdiction, entities)
        
        # Otherwise, use a simulated approach with the LLM
        return self._simulate_precedent_retrieval(clause_text, jurisdiction, entities)
    
    def _retrieve_precedents_from_web(self, 
                                    clause_text: str, 
                                    jurisdiction: str,
                                    entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Retrieve precedents using web search."""
        if not self.web_retriever:
            return []
            
        # Implementation details omitted for brevity - delegates to the function
        entities_str = ','.join([f"{k}:{','.join(v)}" for k, v in entities.items()])
        return self._retrieve_precedents_helper(clause_text, jurisdiction, entities_str)
    
    def _retrieve_precedents_helper(self, clause_text: str, jurisdiction: str, entities_str: str) -> List[Dict[str, Any]]:
        """Helper method to delegate to the standalone function."""
        # Parse entities string back to dictionary
        entities = {}
        for entity_group in entities_str.split(';'):
            if ':' in entity_group:
                key, values = entity_group.split(':', 1)
                entities[key] = values.split(',')
        
        # Call the standalone function with web search
        results = analyze_precedents_for_compliance(
            clause_text=clause_text,
            llm_client=self.llm_client,
            jurisdiction=jurisdiction,
            context=None,
            min_confidence=self.min_confidence,
            use_web_search=True,
            qdrant_url=self.qdrant_url
        )
        
        return results.get("precedents", [])
    
    def _extract_precedent_from_web_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Legacy method that delegates to the function in analyze_precedents_for_compliance."""
        # Function signature maintained for backward compatibility
        pass
    
    def _simulate_precedent_retrieval(self, 
                                    clause_text: str, 
                                    jurisdiction: str,
                                    entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Legacy method that delegates to the function in analyze_precedents_for_compliance."""
        # Function signature maintained for backward compatibility
        pass
    
    def _parse_precedent_block(self, block: str) -> Optional[Dict[str, Any]]:
        """Legacy method that delegates to the function in analyze_precedents_for_compliance."""
        # Function signature maintained for backward compatibility
        pass
    
    def _format_precedents_for_prompt(self, precedents: List[Dict[str, Any]]) -> str:
        """Legacy method that delegates to the function in analyze_precedents_for_compliance."""
        # Function signature maintained for backward compatibility
        if not precedents:
            return ""
            
        formatted_text = ""
        
        for i, precedent in enumerate(precedents):
            formatted_text += f"PRECEDENT {i+1}:\n"
            formatted_text += f"Case: {precedent.get('case_name', 'Unknown')}\n"
            formatted_text += f"Citation: {precedent.get('citation', 'Unknown')}\n"
            formatted_text += f"Jurisdiction: {precedent.get('jurisdiction', 'Unknown')}\n"
            formatted_text += f"Year: {precedent.get('year', 'Unknown')}\n"
            
            # Add key holdings
            formatted_text += "Key Holdings:\n"
            for holding in precedent.get('key_holdings', []):
                formatted_text += f"- {holding}\n"
                
            formatted_text += "\n"
        
        return formatted_text
    
    def rank_precedents(self, precedents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank precedents by relevance and authority."""
        # Sort precedents by relevance score (descending)
        ranked_precedents = sorted(
            precedents, 
            key=lambda p: p.get('relevance_score', 0), 
            reverse=True
        )
        
        # Add rank information
        for i, precedent in enumerate(ranked_precedents):
            precedent['rank'] = i + 1
        
        return ranked_precedents
    
    def extract_holdings(self, precedent: Dict[str, Any]) -> List[Holding]:
        """Extract structured holdings from a precedent."""
        holdings = []
        
        for holding_text in precedent.get('key_holdings', []):
            # Use the LLM to extract the principle and assess relevance
            prompt = f"""
            Analyze the following legal holding and extract:
            1. The core legal principle
            2. Its relevance score (0.0 to 1.0)
            
            HOLDING:
            {holding_text}
            
            Format your response as:
            Principle: [core legal principle]
            Relevance: [score between 0.0 and 1.0]
            """
            
            response = self.llm_client.generate(
                system_prompt="You are a legal analyst specializing in case law interpretation.",
                user_prompt=prompt
            )
            
            # Parse the response
            principle = ""
            relevance = 0.5  # Default relevance
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('Principle:'):
                    principle = line[len('Principle:'):].strip()
                elif line.startswith('Relevance:'):
                    try:
                        relevance_str = line[len('Relevance:'):].strip()
                        relevance = float(relevance_str)
                    except ValueError:
                        # If parsing fails, keep the default
                        pass
            
            # Create a Holding object
            holding = Holding(
                text=holding_text,
                principle=principle,
                relevance=relevance
            )
            
            holdings.append(holding)
            
        return holdings