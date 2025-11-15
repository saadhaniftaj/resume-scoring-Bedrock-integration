from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.processors.ai_matcher import analyze_resume_with_job, analyze_resume_with_job_ai
from typing import Optional, Dict, Any

router = APIRouter()

# ============================================================================
# LEGACY ENDPOINT MODELS (Backward Compatibility)
# ============================================================================

class AnalyzeRequest(BaseModel):
    resume_text: str
    job_description: str

class AnalyzeResponse(BaseModel):
    name: Optional[str]
    location: Optional[str]
    degree: Optional[str]
    work_experience_years: Optional[int]
    work_permit: Optional[str]
    skills: list
    certifications: list
    must_haves_missing: list
    nice_to_haves_missing: list
    match_percentage: float
    summary: Optional[str]
    error: Optional[str]


# ============================================================================
# NEW AI-DRIVEN SCORING ENDPOINT MODELS
# ============================================================================

class CandidateInfo(BaseModel):
    """Candidate information extracted from resume."""
    name: Optional[str] = None
    location: Optional[str] = None
    work_experience_years: Optional[int] = None


class AnalysisBreakdown(BaseModel):
    """Detailed breakdown of the AI scoring analysis."""
    location_match: bool
    must_haves_total: int
    must_haves_met: int
    nice_to_haves_total: int
    nice_to_haves_met: int


class AnalyzeAIResponse(BaseModel):
    """
    Response model for the new AI-driven scoring endpoint.
    
    This model represents the structured output from the AI scoring system,
    which applies complex business rules (location caps, synonym recognition,
    soft skill exclusion, date calculation, etc.).
    """
    match_percentage: int
    candidate_info: CandidateInfo
    analysis_breakdown: AnalysisBreakdown
    summary: str
    error: Optional[str] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_resume(request: AnalyzeRequest):
    """
    LEGACY ENDPOINT: Analyze resume using the original formula-based scoring.
    
    This endpoint is maintained for backward compatibility.
    For new integrations, use /analyze-ai instead.
    """
    try:
        result = analyze_resume_with_job(request.resume_text, request.job_description)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-ai", response_model=AnalyzeAIResponse)
async def analyze_resume_ai(request: AnalyzeRequest):
    """
    NEW AI-DRIVEN SCORING ENDPOINT: Analyze resume using advanced AI business logic.
    
    This endpoint uses a sophisticated AI-driven scoring system that:
    - Enforces location caps (max 5% if location mismatch)
    - Applies zero-tolerance for must-haves (0% if none met)
    - Recognizes technical synonyms (SQL = PostgreSQL/MySQL)
    - Ignores soft skills from scoring calculations
    - Calculates experience from date ranges
    - Adds nice-to-have bonuses (up to 10%)
    
    The AI is instructed via a detailed system prompt to follow strict
    business rules and return a structured JSON response.
    
    Args:
        request: AnalyzeRequest with resume_text and job_description
    
    Returns:
        AnalyzeAIResponse with match percentage, candidate info, analysis breakdown, and summary
    
    Raises:
        HTTPException: If analysis fails
    """
    try:
        result = analyze_resume_with_job_ai(request.resume_text, request.job_description)
        
        # If there's an error in the result, return it with appropriate HTTP error
        if result.get("error"):
            # Still return 200 but with error field populated
            # This allows client to handle AI errors gracefully
            return result
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))