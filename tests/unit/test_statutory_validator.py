# tests/unit/test_statutory_validator.py
import unittest
from unittest.mock import Mock, patch, MagicMock
import uuid

from agents.statutory_validator import StatutoryValidator, Violation, SeverityLevel, Statute


class TestStatutoryValidator(unittest.TestCase):
    """
    Unit tests for the StatutoryValidator component.
    
    The StatutoryValidator is responsible for checking legal clauses against relevant
    statutes and identifying potential violations. It uses a knowledge agent to retrieve
    relevant statutes and an LLM to analyze the clause against those statutes.
    """

    def setUp(self):
        """
        Set up test fixtures.
        
        Creates mock LLM and knowledge agent to isolate the validator component.
        The mock LLM returns a sample violation response, and the mock knowledge agent
        returns a sample statute about labor notice periods.
        """
        # Mock LLM that returns a sample violation in the expected format
        self.mock_llm = Mock()
        self.mock_llm.generate = Mock(return_value="""
        [ISSUE]
        Description: Clause violates minimum notice period requirements
        Severity: HIGH
        References: Labor Code Section 2.3.1
        Reasoning: The clause only provides 7 days notice, but the statute requires at least 14 days.
        [/ISSUE]
        """)
        
        # Mock knowledge agent that returns a relevant statute
        self.mock_knowledge_agent = Mock()
        self.mock_knowledge_agent.find_relevant_statutes = Mock(return_value=[
            {
                "id": "labor-2.3.1",
                "name": "Labor Code",
                "section": "2.3.1",
                "jurisdiction": "US",
                "text": "Employers must provide at least 14 days notice for schedule changes."
            }
        ])
        
        # Initialize the validator with mocked dependencies
        self.validator = StatutoryValidator(
            llm_client=self.mock_llm,
            knowledge_agent=self.mock_knowledge_agent,
            min_confidence=0.7
        )
    
    def test_validate_clause_with_violations(self):
        """
        Test validation that finds violations.
        
        This test verifies that the validator correctly identifies a statutory violation
        when a clause contradicts a statute. The expected output is a list containing
        a single Violation object with details about the violation.
        
        Input: A clause text that violates labor notice requirements
        Expected output: A list with one Violation object with HIGH severity
        """
        # Arrange - Create a clause that violates the 14-day notice requirement
        clause_text = "The employer may change work schedules with 7 days notice."
        clause_id = str(uuid.uuid4())
        
        # Act - Validate the clause against statutes
        violations = self.validator.validate_clause(
            clause_text=clause_text,
            clause_id=clause_id,
            jurisdiction="US"
        )
        
        # Assert - Check that a violation was found with the expected properties
        self.assertEqual(len(violations), 1)
        violation = violations[0]
        self.assertEqual(violation.severity, SeverityLevel.HIGH)
        self.assertEqual(violation.statute_reference, "Labor Code Section 2.3.1")
        self.assertIn("minimum notice period", violation.description)
        self.assertIn("7 days", violation.reasoning)
        self.assertIn("14 days", violation.reasoning)
    
    def test_validate_clause_no_violations(self):
        """
        Test validation that finds no violations.
        
        This test verifies that the validator correctly returns an empty list when
        a clause complies with all relevant statutes.
        
        Input: A clause text that complies with labor notice requirements
        Expected output: An empty list (no violations)
        """
        # Arrange - Mock LLM to return no issues and create a compliant clause
        self.mock_llm.generate = Mock(return_value="[NO_ISSUES]")
        clause_text = "The employer may change work schedules with 14 days notice."
        
        # Act - Validate the compliant clause
        violations = self.validator.validate_clause(
            clause_text=clause_text,
            jurisdiction="US"
        )
        
        # Assert - Check that no violations were found
        self.assertEqual(len(violations), 0)
    
    def test_validate_clause_with_knowledge_agent(self):
        """
        Test that the knowledge agent is used to retrieve statutes.
        
        This test verifies that the validator correctly calls the knowledge agent
        to retrieve relevant statutes for the clause being validated.
        
        Input: A clause text about work schedules
        Expected behavior: Knowledge agent is called with appropriate parameters
        """
        # Arrange - Create a clause about work schedules
        clause_text = "The employer may change work schedules with 7 days notice."
        
        # Act - Validate the clause, which should call the knowledge agent
        self.validator.validate_clause(
            clause_text=clause_text,
            jurisdiction="US"
        )
        
        # Assert - Verify that the knowledge agent was called with correct parameters
        self.mock_knowledge_agent.find_relevant_statutes.assert_called_once()
        call_args = self.mock_knowledge_agent.find_relevant_statutes.call_args[1]
        self.assertEqual(call_args["jurisdiction"], "US")
        self.assertIn("schedule", call_args["query"].lower())
    
    def test_validate_clause_without_knowledge_agent(self):
        """
        Test validation without a knowledge agent.
        
        This test verifies that the validator can still function when no knowledge agent
        is provided, relying solely on the LLM's knowledge.
        
        Input: A clause text with no knowledge agent available
        Expected output: Still able to identify violations using just the LLM
        """
        # Arrange - Create validator without knowledge agent
        validator = StatutoryValidator(
            llm_client=self.mock_llm,
            knowledge_agent=None
        )
        clause_text = "The employer may change work schedules with 7 days notice."
        
        # Act - Validate the clause without knowledge agent
        violations = validator.validate_clause(
            clause_text=clause_text,
            jurisdiction="US"
        )
        
        # Assert - Check that violations can still be identified
        self.assertEqual(len(violations), 1)  # Should still work using just the LLM
    
    def test_normalize_severity(self):
        """
        Test severity normalization.
        
        This test verifies that the validator correctly normalizes severity strings
        to the appropriate SeverityLevel enum values, with unknown values defaulting to MEDIUM.
        
        Input: Various severity strings (HIGH, MEDIUM, LOW, UNKNOWN)
        Expected output: Corresponding SeverityLevel enum values
        """
        # Arrange & Act - Normalize different severity strings
        high = self.validator._normalize_severity("HIGH")
        medium = self.validator._normalize_severity("MEDIUM")
        low = self.validator._normalize_severity("LOW")
        unknown = self.validator._normalize_severity("UNKNOWN")
        
        # Assert - Check that each string is normalized correctly
        self.assertEqual(high, SeverityLevel.HIGH)
        self.assertEqual(medium, SeverityLevel.MEDIUM)
        self.assertEqual(low, SeverityLevel.LOW)
        self.assertEqual(unknown, SeverityLevel.MEDIUM)  # Default to MEDIUM
    
    def test_extract_implications(self):
        """
        Test extraction of implications from reasoning.
        
        This test verifies that the validator correctly extracts implications from
        the reasoning text provided by the LLM.
        
        Input: Reasoning text containing multiple implications
        Expected output: List of extracted implication strings
        """
        # Arrange - Create reasoning text with multiple implications
        reasoning = "This could lead to legal liability. Another implication is increased costs."
        
        # Act - Extract implications from the reasoning
        implications = self.validator._extract_implications(reasoning)
        
        # Assert - Check that both implications were extracted
        self.assertEqual(len(implications), 2)
        self.assertIn("legal liability", implications[0])
        self.assertIn("increased costs", implications[1])


if __name__ == '__main__':
    unittest.main() 