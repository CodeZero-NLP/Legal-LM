from typing import Dict, List, Any


class ResponseParser:
    """
    Parser for structured LLM responses to extract legal analysis information.
    """
    
    def __init__(self):
        """Initialize the response parser."""
        pass
    
    def parse_issues(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response to extract structured issue information.
        
        Expected LLM response format:
        [ISSUE]
        Description: <description>
        Severity: <HIGH|MEDIUM|LOW>
        References: <references>
        Reasoning: <reasoning>
        [/ISSUE]
        
        Args:
            response: Raw response text from the LLM
            
        Returns:
            List[Dict]: List of parsed issues
        """
        issues = []
        
        # Check for "no issues" response
        if "[NO_ISSUES]" in response:
            return []
        
        # Split response into individual issue blocks
        issue_blocks = response.split('[ISSUE]')
        
        for block in issue_blocks:
            if not block.strip() or '[/ISSUE]' not in block:
                continue
            
            # Extract issue content
            content = block.split('[/ISSUE]')[0].strip()
            
            # Initialize issue dictionary with default values
            issue = {
                'description': '',
                'severity': 'MEDIUM',  # Default severity
                'references': [],
                'reasoning': '',
            }
            
            # Parse each field from the content
            current_field = None
            field_content = []
            
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Check for field headers
                if line.endswith(':'):
                    # Save previous field if it exists
                    if current_field and field_content:
                        self._update_issue_field(issue, current_field, field_content)
                        field_content = []
                    
                    current_field = line[:-1].lower()  # Remove colon and convert to lowercase
                else:
                    field_content.append(line)
            
            # Save the last field
            if current_field and field_content:
                self._update_issue_field(issue, current_field, field_content)
            
            # Validate required fields
            if issue['description'] and (issue['reasoning'] or issue['references']):
                issues.append(issue)
        
        return issues
    
    def _update_issue_field(self, issue: Dict[str, Any], field: str, content: List[str]) -> None:
        """
        Update an issue dictionary with parsed field content.
        
        Args:
            issue: Issue dictionary to update
            field: Field name to update
            content: List of content lines for the field
        """
        if field == 'description':
            issue['description'] = ' '.join(content)
        elif field == 'severity':
            # Normalize severity to one of the accepted values
            severity = content[0].upper()
            if severity in ['HIGH', 'MEDIUM', 'LOW']:
                issue['severity'] = severity
        elif field == 'references':
            # Clean and normalize references
            issue['references'] = [ref.strip() for ref in content if ref.strip()]
        elif field == 'reasoning':
            issue['reasoning'] = ' '.join(content)
    
    def parse_statutory_analysis(self, response: str) -> Dict[str, Any]:
        """
        Parse a statutory analysis response.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Dict: Structured statutory analysis
        """
        issues = self.parse_issues(response)
        return {
            "issues": issues,
            "issue_count": len(issues),
            "has_issues": len(issues) > 0,
            "analysis_type": "statutory"
        }
    
    def parse_precedent_analysis(self, response: str) -> Dict[str, Any]:
        """
        Parse a precedent analysis response.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Dict: Structured precedent analysis
        """
        issues = self.parse_issues(response)
        return {
            "issues": issues,
            "issue_count": len(issues),
            "has_issues": len(issues) > 0,
            "analysis_type": "precedent"
        }
    
    def parse_consistency_check(self, response: str) -> Dict[str, Any]:
        """
        Parse a consistency check response.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Dict: Structured consistency analysis
        """
        issues = self.parse_issues(response)
        return {
            "issues": issues,
            "issue_count": len(issues),
            "has_issues": len(issues) > 0,
            "analysis_type": "consistency"
        } 