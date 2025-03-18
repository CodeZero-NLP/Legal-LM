import unittest
from unittest.mock import MagicMock, patch
import json
import uuid

from agents.statutory_validator import StatutoryValidator, Statute, Violation, SeverityLevel
from utils.ollama_client import OllamaClient
from utils.prompt_templates import PromptTemplates
from utils.response_parser import ResponseParser
from utils.confidence_scorer import ConfidenceScorer


class TestStatutoryValidatorIntegration(unittest.TestCase):
    """
    Integration tests for the StatutoryValidator component.
    
    These tests verify that the StatutoryValidator correctly:
    1. Validates clauses against statutory laws
    2. Integrates with the knowledge agent for statute retrieval
    3. Assesses violation severity
    4. Extracts implications from reasoning
    """

    def setUp(self):
        """
        Set up common test fixtures.
        This includes:
        - Mock LLM client to avoid actual LLM calls
        - Mock knowledge agent for statute retrieval
        - Sample clauses and statutes for testing
        """
        # Create a mock LLM client
        self.mock_llm_client = MagicMock()
        
        # Set up default response for the mock LLM
        self.mock_llm_client.generate.return_value = json.dumps({
            "issues": [
                {
                    "description": "Test statutory violation",
                    "severity": "MEDIUM",
                    "reasoning": "This is test reasoning",
                    "references": ["Test Statute 123"],
                    "confidence": 0.85
                }
            ]
        })
        
        # Create a mock knowledge agent
        self.mock_knowledge_agent = MagicMock()
        
        # Set up default response for the mock knowledge agent
        self.mock_knowledge_agent.find_relevant_statutes.return_value = [
            {
                "id": "statute1",
                "name": "Test Statute",
                "section": "123",
                "jurisdiction": "US",
                "text": "This is a test statute text."
            }
        ]
        
        # Create sample clauses for testing
        self.sample_clauses = {
            "valid_clause": "The Tenant shall pay a security deposit of $1,000 upon signing this agreement.",
            "invalid_clause": "The Tenant waives all rights to a security deposit refund under any circumstances."
        }
        
        # Create sample statutes for testing
        self.sample_statutes = [
            Statute(
                id="cal_civ_1950.5",
                name="California Civil Code",
                section="1950.5",
                jurisdiction="California",
                text="Security deposits for residential leases shall not exceed two months' rent for unfurnished units."
            ),
            Statute(
                id="tenant_rights_123",
                name="Tenant Rights Protection Act",
                section="123",
                jurisdiction="US",
                text="Landlords may not require tenants to waive rights to security deposit refunds."
            )
        ]
        
        # Create utility components
        self.prompt_templates = PromptTemplates()
        self.response_parser = ResponseParser()
        self.confidence_scorer = ConfidenceScorer(min_confidence_threshold=0.75)

    def test_validate_clause_with_knowledge_agent(self):
        """
        Test the validate_clause method with a knowledge agent.
        
        This test verifies that:
        1. The validator correctly retrieves statutes from the knowledge agent
        2. It formats the prompt with the retrieved statutes
        3. It processes the LLM response correctly
        4. It returns properly structured Violation objects
        """
        # Configure the mock LLM response
        statutory_response = """
        {
            "issues": [
                {
                    "description": "Clause violates tenant rights protection",
                    "severity": "HIGH",
                    "reasoning": "The clause attempts to waive tenant rights to security deposit refunds, which is prohibited by the Tenant Rights Protection Act",
                    "references": ["Tenant Rights Protection Act § 123"],
                    "confidence": 0.95
                }
            ]
        }
        """
        self.mock_llm_client.generate.return_value = statutory_response
        
        # Configure the mock knowledge agent
        self.mock_knowledge_agent.find_relevant_statutes.return_value = [
            {
                "id": "tenant_rights_123",
                "name": "Tenant Rights Protection Act",
                "section": "123",
                "jurisdiction": "US",
                "text": "Landlords may not require tenants to waive rights to security deposit refunds."
            }
        ]
        
        # Create the validator with our mocks
        validator = StatutoryValidator(
            llm_client=self.mock_llm_client,
            knowledge_agent=self.mock_knowledge_agent,
            min_confidence=0.75
        )
        
        # Test the validator with an invalid clause
        violations = validator.validate_clause(
            clause_text=self.sample_clauses["invalid_clause"],
            clause_id="clause1",
            jurisdiction="US"
        )
        
        # Verify the knowledge agent was called
        self.mock_knowledge_agent.find_relevant_statutes.assert_called_once()
        
        # Verify the LLM was called with the right prompt
        self.mock_llm_client.generate.assert_called_once()
        args, kwargs = self.mock_llm_client.generate.call_args
        self.assertIn("Tenant Rights Protection Act", kwargs["user_prompt"])
        
        # Verify the results
        self.assertIsInstance(violations, list)
        self.assertEqual(len(violations), 1)
        self.assertIsInstance(violations[0], Violation)
        self.assertEqual(violations[0].severity, SeverityLevel.HIGH)
        self.assertEqual(violations[0].statute_reference, "Tenant Rights Protection Act § 123")
        self.assertGreaterEqual(violations[0].confidence, 0.75)

    def test_validate_clause_without_knowledge_agent(self):
        """
        Test the validate_clause method without a knowledge agent.
        
        This test verifies that:
        1. The validator works correctly without a knowledge agent
        2. It formats the prompt properly with just the clause and jurisdiction
        3. It processes the LLM response correctly
        4. It returns properly structured Violation objects
        """
        # Configure the mock LLM response
        statutory_response = """
        {
            "issues": [
                {
                    "description": "Clause likely violates security deposit laws",
                    "severity": "MEDIUM",
                    "reasoning": "Many jurisdictions prohibit clauses that waive tenant rights to security deposit refunds",
                    "references": ["Common tenant protection laws"],
                    "confidence": 0.85
                }
            ]
        }
        """
        self.mock_llm_client.generate.return_value = statutory_response
        
        # Create the validator without a knowledge agent
        validator = StatutoryValidator(
            llm_client=self.mock_llm_client,
            knowledge_agent=None,
            min_confidence=0.75
        )
        
        # Test the validator with an invalid clause
        violations = validator.validate_clause(
            clause_text=self.sample_clauses["invalid_clause"],
            clause_id="clause1",
            jurisdiction="US"
        )
        
        # Verify the LLM was called with the right prompt
        self.mock_llm_client.generate.assert_called_once()
        args, kwargs = self.mock_llm_client.generate.call_args
        self.assertIn(self.sample_clauses["invalid_clause"], kwargs["user_prompt"])
        self.assertIn("US", kwargs["user_prompt"])
        
        # Verify the results
        self.assertIsInstance(violations, list)
        self.assertEqual(len(violations), 1)
        self.assertIsInstance(violations[0], Violation)
        self.assertEqual(violations[0].severity, SeverityLevel.MEDIUM)
        self.assertEqual(violations[0].statute_reference, "Common tenant protection laws")
        self.assertGreaterEqual(violations[0].confidence, 0.75)

    def test_assess_severity(self):
        """
        Test the assess_severity method.
        
        This test verifies that:
        1. The validator correctly assesses violation severity
        2. It applies heuristics based on reasoning content
        3. It normalizes severity levels
        4. It handles both Violation objects and dictionaries
        """
        # Create the validator
        validator = StatutoryValidator(
            llm_client=self.mock_llm_client,
            min_confidence=0.75
        )
        
        # Create test violations with different reasoning
        high_violation = Violation(
            id="v1",
            clause_id="clause1",
            statute_reference="Test Statute 1",
            severity=SeverityLevel.MEDIUM,  # Initial severity
            description="Test violation 1",
            implications=["Implication 1"],
            reasoning="This clause is illegal and creates criminal liability for the landlord.",
            confidence=0.9
        )
        
        medium_violation = Violation(
            id="v2",
            clause_id="clause1",
            statute_reference="Test Statute 2",
            severity=SeverityLevel.LOW,  # Initial severity
            description="Test violation 2",
            implications=["Implication 2"],
            reasoning="This clause is ambiguous and may not comply with regulations.",
            confidence=0.8
        )
        
        # Test with Violation objects
        high_severity = validator.assess_severity(high_violation)
        medium_severity = validator.assess_severity(medium_violation)
        
        # Verify the results
        self.assertEqual(high_severity, SeverityLevel.HIGH)
        self.assertEqual(medium_severity, SeverityLevel.MEDIUM)
        
        # Test with dictionaries
        dict_violation = {
            "severity": "INVALID_LEVEL",  # Invalid severity
            "reasoning": "This is test reasoning"
        }
        
        default_severity = validator.assess_severity(dict_violation)
        self.assertEqual(default_severity, SeverityLevel.MEDIUM)  # Should default to MEDIUM

    def test_get_relevant_statutes(self):
        """
        Test the get_relevant_statutes method.
        
        This test verifies that:
        1. The validator correctly retrieves statutes from the knowledge agent
        2. It converts raw statute data to Statute objects
        3. It handles errors gracefully
        4. It returns an empty list when no knowledge agent is available
        """
        # Configure the mock knowledge agent
        self.mock_knowledge_agent.find_relevant_statutes.return_value = [
            {
                "id": "cal_civ_1950.5",
                "name": "California Civil Code",
                "section": "1950.5",
                "jurisdiction": "California",
                "text": "Security deposits for residential leases shall not exceed two months' rent for unfurnished units.",
                "url": "https://example.com/statutes/cal_civ_1950.5"
            }
        ]
        
        # Create the validator with our mock
        validator = StatutoryValidator(
            llm_client=self.mock_llm_client,
            knowledge_agent=self.mock_knowledge_agent,
            min_confidence=0.75
        )
        
        # Test the get_relevant_statutes method
        clause_context = {
            "text": "The Tenant shall pay a security deposit of $5,000 upon signing this agreement.",
            "jurisdiction": "California"
        }
        
        statutes = validator.get_relevant_statutes(clause_context)
        
        # Verify the knowledge agent was called with the right parameters
        self.mock_knowledge_agent.find_relevant_statutes.assert_called_once_with(
            query=clause_context["text"],
            jurisdiction=clause_context["jurisdiction"]
        )
        
        # Verify the results
        self.assertIsInstance(statutes, list)
        self.assertEqual(len(statutes), 1)
        self.assertIsInstance(statutes[0], Statute)
        self.assertEqual(statutes[0].id, "cal_civ_1950.5")
        self.assertEqual(statutes[0].jurisdiction, "California")
        self.assertEqual(statutes[0].url, "https://example.com/statutes/cal_civ_1950.5")
        
        # Test with no knowledge agent
        validator_no_agent = StatutoryValidator(
            llm_client=self.mock_llm_client,
            knowledge_agent=None,
            min_confidence=0.75
        )
        
        empty_statutes = validator_no_agent.get_relevant_statutes(clause_context)
        self.assertEqual(empty_statutes, [])
        
        # Test error handling
        self.mock_knowledge_agent.find_relevant_statutes.side_effect = Exception("Test error")
        error_statutes = validator.get_relevant_statutes(clause_context)
        self.assertEqual(error_statutes, [])

    def test_extract_implications(self):
        """
        Test the _extract_implications method.
        
        This test verifies that:
        1. The validator correctly extracts implications from reasoning text
        2. It identifies sentences containing implication keywords
        3. It returns a list of implication strings
        4. It handles cases with no clear implications
        """
        # Create the validator
        validator = StatutoryValidator(
            llm_client=self.mock_llm_client,
            min_confidence=0.75
        )
        
        # Test with reasoning containing implications
        reasoning_with_implications = "This clause violates tenant protection laws. The implication is that the landlord could face penalties. Another consequence is potential litigation from tenants."
        
        implications = validator._extract_implications(reasoning_with_implications)
        
        # Verify the results
        self.assertIsInstance(implications, list)
        self.assertEqual(len(implications), 2)
        self.assertIn("The implication is that the landlord could face penalties", implications)
        self.assertIn("Another consequence is potential litigation from tenants", implications)
        
        # Test with reasoning not containing clear implications
        reasoning_without_implications = "This clause appears to be compliant with relevant statutes."
        
        generic_implications = validator._extract_implications(reasoning_without_implications)
        
        # Verify a generic implication is returned
        self.assertIsInstance(generic_implications, list)
        self.assertEqual(len(generic_implications), 1)
        self.assertTrue(generic_implications[0].startswith("Potential compliance issue"))


if __name__ == '__main__':
    unittest.main() 