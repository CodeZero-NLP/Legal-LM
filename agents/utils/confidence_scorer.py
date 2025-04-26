from typing import Dict, Any, List

class ConfidenceScorer:
    """
    Evaluates the reliability and confidence of LLM-generated legal analyses.
    """
    
    def __init__(self, min_confidence_threshold: float = 0.75):
        """
        Initialize the confidence scorer.
        
        Args:
            min_confidence_threshold: Minimum confidence threshold for valid issues
        """
        self.min_confidence_threshold = min_confidence_threshold
    
    def score_issue(self, issue: Dict[str, Any]) -> float:
        """
        Calculate a confidence score for an individual issue.
        
        Args:
            issue: Parsed issue dictionary
            
        Returns:
            float: Confidence score between 0 and 1
        """
        score = 0.0
        
        # Basic presence checks
        if issue.get('description', ''):
            score += 0.3
        if issue.get('reasoning', ''):
            score += 0.3
        if issue.get('references', []):
            score += 0.2
        if issue.get('severity', 'MEDIUM') != 'MEDIUM':  # If severity was explicitly set
            score += 0.2
        
        # Quality checks
        description = issue.get('description', '')
        reasoning = issue.get('reasoning', '')
        references = issue.get('references', [])
        
        if len(description) > 20:  # Reasonable description length
            score += 0.1
        if len(reasoning) > 50:    # Detailed reasoning
            score += 0.1
        if len(references) > 1:    # Multiple references
            score += 0.1
        
        # Normalize to 0-1 range
        return min(1.0, score)
    
    def score_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score an entire analysis result and filter low-confidence issues.
        
        Args:
            analysis: Analysis result containing issues
            
        Returns:
            Dict: Analysis with confidence scores and filtered issues
        """
        issues = analysis.get("issues", [])
        scored_issues = []
        
        for issue in issues:
            confidence = self.score_issue(issue)
            issue["confidence"] = confidence
            if confidence >= self.min_confidence_threshold:
                scored_issues.append(issue)
        
        analysis["scored_issues"] = scored_issues
        analysis["valid_issue_count"] = len(scored_issues)
        analysis["average_confidence"] = sum(issue["confidence"] for issue in scored_issues) / len(scored_issues) if scored_issues else 0
        
        return analysis
    
    def get_high_confidence_issues(self, analysis: Dict[str, Any], threshold: float = None) -> List[Dict[str, Any]]:
        """
        Extract only high-confidence issues from an analysis.
        
        Args:
            analysis: Analysis result containing scored issues
            threshold: Optional custom threshold (uses instance default if None)
            
        Returns:
            List[Dict]: List of high-confidence issues
        """
        if threshold is None:
            threshold = self.min_confidence_threshold
        
        if "scored_issues" in analysis:
            return [issue for issue in analysis["scored_issues"] if issue.get("confidence", 0) >= threshold]
        else:
            # Score issues if not already scored
            self.score_analysis(analysis)
            return [issue for issue in analysis["scored_issues"] if issue.get("confidence", 0) >= threshold] 