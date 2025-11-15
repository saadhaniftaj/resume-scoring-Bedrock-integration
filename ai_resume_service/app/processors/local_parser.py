"""
Local Resume Parser Module

Provides basic resume parsing using spaCy and regex patterns.
"""

import re
import spacy
from typing import Dict, List
from app.config.app_config import ModelConfig

class ResumeParser:
    """Basic resume parser for contractor-specific information."""
    
    def __init__(self):
        """Initialize parser with configured spaCy model."""
        self.nlp = spacy.load(ModelConfig.SPACY_MODEL)
        self._init_skills_db()
    
    def _init_skills_db(self):
        """Contractor-specific skills"""
        self.tech_skills = ["Python", "AWS", "Docker"]
        self.contract_skills = ["1099", "W2", "C2C", "Hourly"]
    
    def parse(self, text: str) -> Dict:
        return {
            "education": self._parse_education(text),
            "skills": self._parse_skills(text),
            "experience": self._parse_experience(text),
            "contract_type": self._parse_contract_type(text)
        }
    
    def _parse_education(self, text: str) -> str:
        """Simple degree extraction"""
        degrees = {
            "BSc": r"B\.?Sc|\bBachelor",
            "MSc": r"M\.?Sc|\bMaster",
            "PhD": r"Ph\.?D|\bDoctorate"
        }
        for degree, pattern in degrees.items():
            if re.search(pattern, text, re.IGNORECASE):
                return degree
        return "Not Specified"
    
    def _parse_skills(self, text: str) -> List[str]:
        """Basic skill matching"""
        found = set()
        for skill in self.tech_skills + self.contract_skills:
            if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE):
                found.add(skill)
        return sorted(found)
    
    def _parse_experience(self, text: str) -> float:
        """Extract years of experience"""
        # 1. Look for explicit year mentions
        matches = re.findall(r"(\d+)\s+years?", text, re.IGNORECASE)
        if matches:
            return float(max(map(int, matches)))
        
        # 2. Fallback: Count experience keywords
        exp_keywords = ["experience", "exp", "worked"]
        keyword_count = sum(1 for kw in exp_keywords if kw in text.lower())
        return float(keyword_count)  # 1 year per keyword found
    
    def _parse_contract_type(self, text: str) -> str:
        """Detect contractor type"""
        text_lower = text.lower()
        if "1099" in text_lower:
            return "1099"
        elif "w2" in text_lower:
            return "W2"
        elif "c2c" in text_lower:
            return "C2C"
        return "Unknown"