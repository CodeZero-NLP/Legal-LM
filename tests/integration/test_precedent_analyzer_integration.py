import unittest
from unittest.mock import MagicMock, patch
import json
import uuid

from agents.precedent_analyzer import PrecedentAnalyzer, PrecedentMatch, Holding, RelevanceLevel
from utils.ollama_client import OllamaClient
from utils.prompt_templates import PromptTemplates
from utils.response_parser import ResponseParser
from utils.confidence_scorer import ConfidenceScorer
from utils.blackstone import BlackstoneNER


class TestPrecedentAnalyzerIntegration(unittest.TestCase):
    """
    Integration tests for the PrecedentAnalyzer component.
    
    These tests verify that the PrecedentAnalyzer correctly:
    1. Retrieves and analyzes legal precedents
    2. Identifies conflicts with case law
    3. Extracts and structures holdings from precedents
    4. Ranks precedents by relevance
    """

    def setUp(self):
        """
        Set up common test fixtures.
        This includes:
        - Mock LLM client to avoid actual LLM calls
        - Mock NER component for legal entity extraction
        - Sample clauses and precedents for testing
        """
        # Create a mock LLM client
        self.mock_llm_client = MagicMock()
        
        # Set up default response for the mock LLM
        self.mock_llm_client.generate.return_value = json.dumps({
            "issues": [
                {
                    "description": "Test precedent issue",
                    "severity": "MEDIUM",
                    "reasoning": "This is test reasoning",
                    "references": ["Test Case v. Other Case"],
                    "confidence": 0.85
                }
            ]
        })
        
        # Create a mock NER component
        self.mock_ner = MagicMock()
        self.mock_ner.extract_entities.return_value = [
            ("security deposit", "PROVISION"),
            ("landlord", "PARTY"),
            ("tenant", "PARTY")
        ]
        
        # Create sample clauses for testing
        self.sample_clauses = {
            "valid_clause": "The Tenant shall pay a security deposit of $1,000 upon signing this agreement.",
            "invalid_clause": "The security deposit may be withheld at the Landlord's discretion without any justification."
        }
        
        # Create sample precedents for testing
        self.sample_precedents = [
            {
                "case_name": "Smith v. Jones",
                "citation": "123 Cal.App.4th 456",
                "jurisdiction": "California",
                "year": 2018,
                "key_holdings": ["Security deposit clauses must specify conditions for withholding funds"],
                "relevance_score": 0.9
            },
            {
                "case_name": "Tenant Association v. Landlord Corp",
                "citation": "234 F.3d 567",
                "jurisdiction": "US",
                "year": 2015,
                "key_holdings": ["Arbitrary withholding of security deposits violates tenant rights"],
                "relevance_score": 0.8
            }
        ]
        
        # Create utility components
        self.prompt_templates = PromptTemplates()
        self.response_parser = ResponseParser()
        self.confidence_scorer = ConfidenceScorer(min_confidence_threshold=0.75)

    def test_analyze_precedents_with_simulated_retrieval(self):
        """
        Test the analyze_precedents method with simulated precedent retrieval.
        
        This test verifies that:
        1. The analyzer correctly simulates precedent retrieval when web search is disabled
        2. It formats the prompt with the retrieved precedents
        3. It processes the LLM response correctly
        4. It returns properly structured analysis results
        """
        # Configure the mock LLM response for precedent analysis
        precedent_response = """
        {
            "issues": [
                {
                    "description": "Clause conflicts with established case law on security deposits",
                    "severity": "HIGH",
                    "reasoning": "The clause allows arbitrary withholding of security deposits, which conflicts with Smith v. Jones (2018) that requires specific conditions for withholding",
                    "references": ["Smith v. Jones, 123 Cal.App.4th 456 (2018)"],
                    "confidence": 0.92
                }
            ],
            "precedents": [
                {
                    "case_name": "Smith v. Jones",
                    "citation": "123 Cal.App.4th 456",
                    "jurisdiction": "California",
                    "year": 2018,
                    "key_holdings": ["Security deposit clauses must specify conditions for withholding funds"],
                    "relevance_score": 0.9
                }
            ]
        }
        """
        self.mock_llm_client.generate.return_value = precedent_response
        
        # Create the analyzer with our mocks
        analyzer = PrecedentAnalyzer(
            llm_client=self.mock_llm_client,
            min_confidence=0.75,
            use_web_search=False
        )
        
        # Replace the NER component with our mock
        analyzer.ner = self.mock_ner
        
        # Mock the _simulate_precedent_retrieval method to return test data
        analyzer._simulate_precedent_retrieval = MagicMock(return_value=self.sample_precedents)
        
        # Test the analyzer with a sample clause
        clause_text = self.sample_clauses["invalid_clause"]
        result = analyzer.analyze_precedents(
            clause_text=clause_text,
            jurisdiction="California"
        )
        
        # Verify the NER was called to extract entities
        self.mock_ner.extract_entities.assert_called_once_with(clause_text)
        
        # Verify the LLM was called with the right prompt
        self.mock_llm_client.generate.assert_called_once()
        args, kwargs = self.mock_llm_client.generate.call_args
        self.assertIn("Smith v. Jones", kwargs["user_prompt"])
        
        # Verify the results
        self.assertIsInstance(result, dict)
        self.assertTrue("issues" in result)
        self.assertTrue("precedents" in result)
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["severity"], "HIGH")
        self.assertEqual(result["issues"][0]["description"], "Clause conflicts with established case law on security deposits")
        self.assertTrue(result["has_issues"])
        self.assertEqual(result["precedent_count"], 2)  # Should match our sample precedents

    @patch('agents.precedent_analyzer.WebContentRetriever')
    def test_analyze_precedents_with_web_search(self, mock_web_retriever_class):
        """
        Test the analyze_precedents method with web search.
        
        This test verifies that:
        1. The analyzer correctly uses web search when enabled
        2. It processes web search results into structured precedents
        3. It formats the prompt with the retrieved precedents
        4. It returns properly structured analysis results
        """
        # Configure the mock web retriever
        mock_web_retriever = MagicMock()
        mock_web_retriever_class.return_value = mock_web_retriever
        
        # Set up web search results
        mock_web_retriever.search_in_qdrant.return_value = [
            {
                "content": "In Smith v. Jones (2018), the court ruled that security deposit clauses must specify the conditions for withholding funds.",
                "url": "https://example.com/case1",
                "score": 0.95
            },
            {
                "content": "Tenant Association v. Landlord Corp (2015) established that arbitrary withholding of security deposits violates tenant rights.",
                "url": "https://example.com/case2",
                "score": 0.85
            }
        ]
        
        # Configure the mock LLM responses for precedent extraction and analysis
        self.mock_llm_client.generate.side_effect = [
            # First call - extract precedent from web result 1
            """
            Case Name: Smith v. Jones
            Citation: 123 Cal.App.4th 456
            Jurisdiction: California
            Year: 2018
            Key Holdings: Security deposit clauses must specify conditions for withholding funds
            """,
            # Second call - extract precedent from web result 2
            """
            Case Name: Tenant Association v. Landlord Corp
            Citation: 234 F.3d 567
            Jurisdiction: US
            Year: 2015
            Key Holdings: Arbitrary withholding of security deposits violates tenant rights
            """,
            # Third call - analyze precedents
            """
            {
                "issues": [
                    {
                        "description": "Clause conflicts with established case law on security deposits",
                        "severity": "HIGH",
                        "reasoning": "The clause allows arbitrary withholding of security deposits, which conflicts with multiple precedents",
                        "references": ["Smith v. Jones", "Tenant Association v. Landlord Corp"],
                        "confidence": 0.95
                    }
                ]
            }
            """
        ]
        
        # Create the analyzer with web search enabled
        analyzer = PrecedentAnalyzer(
            llm_client=self.mock_llm_client,
            min_confidence=0.75,
            use_web_search=True
        )
        
        # Replace the NER component with our mock
        analyzer.ner = self.mock_ner
        
        # Test the analyzer with a sample clause
        clause_text = self.sample_clauses["invalid_clause"]
        result = analyzer.analyze_precedents(
            clause_text=clause_text,
            jurisdiction="California"
        )
        
        # Verify the web retriever was initialized and used
        mock_web_retriever_class.assert_called_once()
        mock_web_retriever.search_in_qdrant.assert_called()
        
        # Verify the LLM was called multiple times
        self.assertGreaterEqual(self.mock_llm_client.generate.call_count, 3)
        
        # Verify the results
        self.assertIsInstance(result, dict)
        self.assertTrue("issues" in result)
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["severity"], "HIGH")
        self.assertTrue(result["has_issues"])

    def test_rank_precedents(self):
        """
        Test the rank_precedents method.
        
        This test verifies that:
        1. The analyzer correctly ranks precedents by relevance
        2. It adds rank information to each precedent
        3. The most relevant precedents are ranked highest
        """
        # Create the analyzer
        analyzer = PrecedentAnalyzer(
            llm_client=self.mock_llm_client,
            min_confidence=0.75
        )
        
        # Create test precedents with varying relevance scores
        test_precedents = [
            {
                "case_name": "Case A",
                "citation": "123 F.3d 456",
                "jurisdiction": "US",
                "year": 2020,
                "relevance_score": 0.7
            },
            {
                "case_name": "Case B",
                "citation": "234 F.3d 567",
                "jurisdiction": "US",
                "year": 2018,
                "relevance_score": 0.9
            },
            {
                "case_name": "Case C",
                "citation": "345 F.3d 678",
                "jurisdiction": "US",
                "year": 2015,
                "relevance_score": 0.8
            }
        ]
        
        # Rank the precedents
        ranked_precedents = analyzer.rank_precedents(test_precedents)
        
        # Verify the results
        self.assertEqual(len(ranked_precedents), 3)
        self.assertEqual(ranked_precedents[0]["case_name"], "Case B")  # Highest relevance
        self.assertEqual(ranked_precedents[0]["rank"], 1)
        self.assertEqual(ranked_precedents[1]["case_name"], "Case C")  # Second highest
        self.assertEqual(ranked_precedents[1]["rank"], 2)
        self.assertEqual(ranked_precedents[2]["case_name"], "Case A")  # Lowest relevance
        self.assertEqual(ranked_precedents[2]["rank"], 3)

    def test_extract_holdings(self):
        """
        Test the extract_holdings method.
        
        This test verifies that:
        1. The analyzer correctly extracts structured holdings from precedents
        2. It uses the LLM to identify core legal principles
        3. It assesses the relevance of each holding
        4. It returns properly structured Holding objects
        """
        # Configure the mock LLM response for holding analysis
        holding_response = """
        Principle: Security deposits must have specific conditions for withholding
        Relevance: 0.85
        """
        self.mock_llm_client.generate.return_value = holding_response
        
        # Create the analyzer
        analyzer = PrecedentAnalyzer(
            llm_client=self.mock_llm_client,
            min_confidence=0.75
        )
        
        # Create a test precedent
        test_precedent = {
            "case_name": "Smith v. Jones",
            "citation": "123 Cal.App.4th 456",
            "jurisdiction": "California",
            "year": 2018,
            "key_holdings": [
                "Security deposit clauses must specify conditions for withholding funds",
                "Landlords must provide itemized lists of deductions"
            ]
        }
        
        # Extract holdings
        holdings = analyzer.extract_holdings(test_precedent)
        
        # Verify the LLM was called for each holding
        self.assertEqual(self.mock_llm_client.generate.call_count, 2)
        
        # Verify the results
        self.assertEqual(len(holdings), 2)
        self.assertIsInstance(holdings[0], Holding)
        self.assertEqual(holdings[0].text, "Security deposit clauses must specify conditions for withholding funds")
        self.assertEqual(holdings[0].principle, "Security deposits must have specific conditions for withholding")
        self.assertEqual(holdings[0].relevance, 0.85)


if __name__ == '__main__':
    unittest.main() 