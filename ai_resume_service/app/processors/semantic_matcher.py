"""
Semantic Matcher Module

Provides semantic similarity matching using sentence transformers.
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from typing import List, Optional
from app.config.app_config import ScoringConfig

# Import model factory (no more module-level instantiation!)
from app.processors.model_factory import get_semantic_model

def find_missing_must_haves(
    requirements: List[str],
    candidate_items: List[str],
    threshold: float = None,
    log_scores: bool = False
) -> List[str]:
    """
    Find requirements that are missing from candidate items.
    
    Args:
        requirements: List of job requirements
        candidate_items: List of candidate skills/experience
        threshold: Similarity threshold (uses config default if None)
        log_scores: Whether to print similarity scores
    
    Returns:
        List of missing requirements
    """
    if not requirements:
        return []
    
    # Use configured threshold if not provided
    if threshold is None:
        threshold = ScoringConfig.SEMANTIC_MATCHER_THRESHOLD
    
    candidate_text = " ".join(candidate_items).lower()
    missing = []
    
    for req in requirements:
        req_lower = req.lower()
        
        # Special handling for degree requirements
        if any(term in req_lower for term in ["ph.d.", "phd", "doctorate"]):
            if has_phd_any_field(candidate_text):
                continue
                
        # Special handling for experience requirements
        if "years" in req_lower and "experience" in req_lower:
            if meets_experience_requirement(req, candidate_items):
                continue
                
        # Special handling for software/tools
        if is_software_requirement(req):
            if software_is_mentioned(req, candidate_items):
                continue
                
        # Normal semantic matching for other requirements
        model = get_semantic_model()
        req_emb = model.encode([clean_text(req)])
        item_embs = model.encode([clean_text(item) for item in candidate_items])
        similarities = cosine_similarity(req_emb, item_embs)[0]
        max_score = np.max(similarities) if similarities.size > 0 else 0
        
        if log_scores:
            print(f"'{req}' vs candidate items: {max_score:.3f}")
            
        if max_score < threshold:
            missing.append(req)
    
    return missing

def has_phd_any_field(text: str) -> bool:
    """Check if text mentions any PhD qualification"""
    return any(term in text for term in ["ph.d.", "phd", "doctorate"])

def meets_experience_requirement(requirement: str, items: List[str]) -> bool:
    """Check if experience requirement is met"""
    year_match = re.search(r"(\d+)\+? years?", requirement.lower())
    if not year_match:
        return False
    
    req_years = int(year_match.group(1))
    
    for item in items:
        item_years = re.findall(r"(\d+)\+? years?", item.lower())
        for years in item_years:
            if int(years) >= req_years:
                return True
    return False

def is_software_requirement(text: str) -> bool:
    """Check if requirement is for specific software/tool"""
    software_keywords = ["proficiency with", "experience with", "knowledge of"]
    return any(keyword in text.lower() for keyword in software_keywords)

def software_is_mentioned(requirement: str, items: List[str]) -> bool:
    """Check if software/tool is mentioned in any form"""
    tools = extract_tools_from_requirement(requirement)
    if not tools:
        return False
        
    candidate_text = " ".join(items).lower()
    return any(tool.lower() in candidate_text for tool in tools)

def extract_tools_from_requirement(text: str) -> List[str]:
    """Extract tool names from requirement"""
    tools = []
    # Match patterns like "Proficiency with X, Y, and Z"
    matches = re.findall(r"(?:with|in|using)\s+([A-Za-z0-9,\s]+)", text, re.IGNORECASE)
    for match in matches:
        tools.extend([t.strip() for t in re.split(r",|\band\b", match)])
    return [t for t in tools if t]

def clean_text(text: str) -> str:
    """Basic text cleaning for semantic matching"""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s+./-]", "", text)  # Keep technical characters
    return " ".join(text.split())