import unittest
from unittest.mock import Mock, patch, MagicMock
import uuid

from agents.precedent_analyzer import PrecedentAnalyzer, PrecedentMatch, Holding, RelevanceLevel


class TestPrecedentAnalyzer(unittest.TestCase):
    """
    Unit tests for the PrecedentAnalyzer component.
    
    The PrecedentAnalyzer is responsible for analyzing legal clauses against relevant
    case law and precedents. It extracts legal entities from clauses, retrieves relevant
    precedents, and identifies potential conflicts with established case law.
    """

    def setUp(self):
        """
        Set up test fixtures.
        
        Creates mock dependencies including:
        - LLM that returns a sample issue related to a precedent
        - NER extractor that identifies case names and legal concepts
        - Web retriever that returns relevant case summaries
        """
        # Mock LLM that returns a sample issue in the expected format
        self.mock_llm = Mock()
        self.mock_llm.generate = Mock(return_value="""
        [ISSUE]
        Description: Clause conflicts with precedent in Smith v. Jones (2018)
        Severity: MEDIUM
        References: Smith v. Jones, 123 F.3d 456 (9th Cir. 2018)
        Reasoning: The clause attempts to limit liability in a way that was rejected in Smith v. Jones, where the court held that such limitations are unenforceable for gross negligence.
        [/ISSUE]
        """)
        
        # Mock NER extractor that identifies legal entities
        self.mock_blackstone = Mock()
        self.mock_blackstone.extract_entities = Mock(return_value=[
            ("Smith v. Jones", "CASENAME"),
            ("gross negligence", "LEGAL_CONCEPT")
        ])
        
        # Mock web retriever that returns case summaries
        self.mock_web_retriever = Mock()
        self.mock_web_retriever.search = Mock(return_value=[
            {
                "title": "Smith v. Jones - Case Summary",
                "content": "The court held that liability limitations for gross negligence are unenforceable.",
                "url": "https://example.com/cases/smith-v-jones",
                "score": 0.95
            }
        ])
        
        # Create the analyzer with mocked dependencies
        self.analyzer = PrecedentAnalyzer(
            llm_client=self.mock_llm,
            min_confidence=0.7
        )
        self.analyzer.ner_extractor = self.mock_blackstone
        self.analyzer.web_retriever = self.mock_web_retriever
    
    def test_analyze_precedents_with_issues(self):
        """
        Test precedent analysis that finds issues.
        
        This test verifies that the analyzer correctly identifies conflicts between
        a clause and relevant case law precedents.
        
        Input: A clause attempting to limit liability for gross negligence
        Expected output: Analysis dict with issues related to Smith v. Jones precedent
        """
        # Arrange - Create a clause that conflicts with the precedent
        clause_text = "Company shall not be liable for any damages, including gross negligence."
        
        # Act - Analyze the clause against precedents
        analysis = self.analyzer.analyze_precedents(
            clause_text=clause_text,
            jurisdiction="US"
        )
        
        # Assert - Check that issues were found with expected properties
        self.assertTrue(analysis["has_issues"])
        self.assertEqual(analysis["issue_count"], 1)
        
        issue = analysis["issues"][0]
        self.assertIn("Smith v. Jones", issue["description"])
        self.assertEqual(issue["severity"], "MEDIUM")
        self.assertIn("Smith v. Jones", issue["references"][0])
        self.assertIn("gross negligence", issue["reasoning"])
    
    def test_analyze_precedents_no_issues(self):
        """
        Test precedent analysis that finds no issues.
        
        This test verifies that the analyzer correctly returns no issues when
        a clause does not conflict with any relevant precedents.
        
        Input: A clause about insurance coverage (no conflicts)
        Expected output: Analysis dict with no issues
        """
        # Arrange - Mock LLM to return no issues and create a non-conflicting clause
        self.mock_llm.generate = Mock(return_value="[NO_ISSUES]")
        clause_text = "Company shall maintain appropriate insurance coverage."
        
        # Act - Analyze the non-conflicting clause
        analysis = self.analyzer.analyze_precedents(
            clause_text=clause_text,
            jurisdiction="US"
        )
        
        # Assert - Check that no issues were found
        self.assertFalse(analysis["has_issues"])
        self.assertEqual(analysis["issue_count"], 0)
        self.assertEqual(len(analysis["issues"]), 0)
    
    def test_extract_legal_entities(self):
        """
        Test extraction of legal entities from clause text.
        
        This test verifies that the analyzer correctly extracts legal entities
        (case names, legal concepts) from the clause text using the NER extractor.
        
        Input: A clause mentioning a case name and legal concept
        Expected output: List of tuples containing entities and their types
        """
        # Arrange - Create a clause with legal entities
        clause_text = "As established in Smith v. Jones, liability for gross negligence cannot be limited."
        
        # Act - Extract legal entities from the clause
        entities = self.analyzer.extract_legal_entities(clause_text)
        
        # Assert - Check that both entities were extracted correctly
        self.assertEqual(len(entities), 2)
        self.assertIn(("Smith v. Jones", "CASENAME"), entities)
        self.assertIn(("gross negligence", "LEGAL_CONCEPT"), entities)
    
    def test_retrieve_precedents(self):
        """
        Test retrieval of precedents.
        
        This test verifies that the analyzer correctly retrieves relevant precedents
        based on the extracted legal entities using the web retriever.
        
        Input: List of legal entities (case name and legal concept)
        Expected output: List of precedent dictionaries with title, content, and URL
        """
        # Arrange - Create a list of legal entities
        entities = [("Smith v. Jones", "CASENAME"), ("gross negligence", "LEGAL_CONCEPT")]
        
        # Act - Retrieve precedents for the entities
        precedents = self.analyzer.retrieve_precedents(entities, "US")
        
        # Assert - Check that the precedent was retrieved with expected properties
        self.assertEqual(len(precedents), 1)
        precedent = precedents[0]
        self.assertEqual(precedent["title"], "Smith v. Jones - Case Summary")
        self.assertIn("gross negligence", precedent["content"])
        self.assertEqual(precedent["url"], "https://example.com/cases/smith-v-jones")
    
    def test_simulate_precedent_retrieval(self):
        """
        Test simulation of precedent retrieval when web search is disabled.
        
        This test verifies that the analyzer can simulate precedent retrieval using
        the LLM when web search is disabled, providing synthetic precedents.
        
        Input: List of legal entities with web search disabled
        Expected output: List of simulated precedent dictionaries
        """
        # Arrange - Create a list of legal entities
        entities = [("Smith v. Jones", "CASENAME"), ("gross negligence", "LEGAL_CONCEPT")]
        
        # Create analyzer with web search disabled
        analyzer = PrecedentAnalyzer(
            llm_client=self.mock_llm,
            min_confidence=0.7,
            use_web_search=False
        )
        
        # Act - Simulate precedent retrieval
        precedents = analyzer.simulate_precedent_retrieval(entities, "US")
        
        # Assert - Check that simulated precedents were generated with expected structure
        self.assertGreater(len(precedents), 0)
        precedent = precedents[0]
        self.assertIn("title", precedent)
        self.assertIn("content", precedent)
        self.assertIn("key_holdings", precedent)
    
    def test_extract_holdings(self):
        """
        Test extraction of holdings from a precedent.
        
        This test verifies that the analyzer correctly extracts structured holdings
        from a precedent, including the principle and relevance score for each holding.
        
        Input: A precedent dictionary with key holdings
        Expected output: List of Holding objects with text, principle, and relevance
        """
        # Arrange - Create a precedent with key holdings
        precedent = {
            "title": "Smith v. Jones",
            "key_holdings": [
                "Liability limitations for gross negligence are unenforceable.",
                "Parties may limit liability for ordinary negligence."
            ]
        }
        
        # Mock the LLM response for holdings
        self.mock_llm.generate = Mock(side_effect=[
            "Principle: Gross negligence liability cannot be limited\nRelevance: 0.9",
            "Principle: Ordinary negligence liability can be limited\nRelevance: 0.7"
        ])
        
        # Act - Extract holdings from the precedent
        holdings = self.analyzer.extract_holdings(precedent)
        
        # Assert - Check that both holdings were extracted with correct properties
        self.assertEqual(len(holdings), 2)
        
        self.assertEqual(holdings[0].text, "Liability limitations for gross negligence are unenforceable.")
        self.assertEqual(holdings[0].principle, "Gross negligence liability cannot be limited")
        self.assertEqual(holdings[0].relevance, 0.9)
        
        self.assertEqual(holdings[1].text, "Parties may limit liability for ordinary negligence.")
        self.assertEqual(holdings[1].principle, "Ordinary negligence liability can be limited")
        self.assertEqual(holdings[1].relevance, 0.7)


if __name__ == '__main__':
    unittest.main() 