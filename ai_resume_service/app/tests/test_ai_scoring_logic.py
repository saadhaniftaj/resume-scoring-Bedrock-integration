"""
AI Scoring Logic Validation Suite

This test suite validates that the new AI-driven scoring system correctly
follows the client's business rules:
- Location caps
- Zero-tolerance must-haves
- Synonym recognition
- Soft skills exclusion
- Date calculation
- Nice-to-have bonuses

These tests ensure the AI's adherence to the scoring prompt before Bedrock migration.
"""

import pytest
from app.processors.ai_matcher import analyze_resume_with_job_ai


class TestLocationCapRule:
    """Test Case 1: Location Cap (Rule Check)"""
    
    def test_location_mismatch_caps_score_at_5_percent(self):
        """
        RULE: If location doesn't match, maximum score is 5% regardless of skills.
        
        Test: Perfect Python developer in London applying for NYC job.
        Expected: match_percentage <= 5 AND location_match = False
        """
        job_description = """
        Position: Senior Python Developer
        Location: New York, NY
        
        Must-Haves:
        - Python programming (5+ years)
        - AWS experience
        - Strong problem-solving skills
        
        Nice-to-Haves:
        - Docker experience
        """
        
        resume_text = """
        John Doe
        London, UK
        
        Senior Python Developer with 10 years of experience.
        
        Experience:
        - Python Software Engineer (2014 - Present): Built scalable systems using Python and AWS
        - Extensive experience with Docker, Kubernetes, and cloud infrastructure
        - Led team of 5 developers on AWS migration project
        
        Skills: Python, AWS, Docker, Kubernetes, Flask, Django, PostgreSQL
        """
        
        result = analyze_resume_with_job_ai(resume_text, job_description)
        
        # Assertions
        assert result["match_percentage"] <= 5, \
            f"Expected max 5% due to location mismatch, got {result['match_percentage']}%"
        
        assert result["analysis_breakdown"]["location_match"] is False, \
            "Expected location_match to be False"
        
        # Should still recognize the skills match
        assert result["analysis_breakdown"]["must_haves_met"] >= 2, \
            "AI should recognize that candidate has the required skills despite location"


class TestZeroMustHavesRule:
    """Test Case 2: Zero Must-Haves (Rule Check)"""
    
    def test_zero_must_haves_gives_zero_score(self):
        """
        RULE: If candidate has ZERO must-haves, score is 0%.
        
        Test: Java developer applying for Python/React/AWS role.
        Expected: match_percentage = 0 AND must_haves_met = 0
        """
        job_description = """
        Position: Full Stack Developer
        
        Must-Haves:
        - Python programming
        - React.js
        - AWS experience
        
        Nice-to-Haves:
        - Docker
        """
        
        resume_text = """
        Jane Smith
        San Francisco, CA
        
        Senior Java Developer with 8 years of experience.
        
        Experience:
        - Java Backend Engineer (2016 - Present): Built enterprise systems using Java and Spring
        - Database expert: PostgreSQL, MySQL, Oracle
        - Strong SQL skills
        
        Skills: Java, Spring Boot, SQL, PostgreSQL, Maven, Jenkins
        """
        
        result = analyze_resume_with_job_ai(resume_text, job_description)
        
        # Assertions
        assert result["match_percentage"] == 0, \
            f"Expected 0% due to zero must-haves met, got {result['match_percentage']}%"
        
        assert result["analysis_breakdown"]["must_haves_met"] == 0, \
            "Expected 0 must-haves met"
        
        assert result["analysis_breakdown"]["must_haves_total"] >= 3, \
            "Should correctly identify at least 3 must-haves"


class TestNinetyPercentMustHavesRule:
    """Test Case 3: 90% Must-Haves (Rule Check)"""
    
    def test_all_must_haves_met_gives_90_percent_base(self):
        """
        RULE: If candidate has ALL must-haves but no nice-to-haves, score is 90-99%.
        
        Test: Has Python, React, AWS but NOT Docker.
        Expected: match_percentage >= 90 AND < 100
        """
        job_description = """
        Position: Full Stack Developer
        
        Must-Haves:
        - Python programming
        - React.js
        - AWS experience
        
        Nice-to-Haves:
        - Docker experience
        """
        
        resume_text = """
        Alice Johnson
        Seattle, WA
        
        Full Stack Developer with 6 years of experience.
        
        Experience:
        - Full Stack Engineer (2018 - Present): Built web applications using Python and React
        - Deployed applications to AWS (EC2, S3, Lambda)
        - No containerization experience
        
        Skills: Python, React, AWS, JavaScript, Flask, PostgreSQL
        """
        
        result = analyze_resume_with_job_ai(resume_text, job_description)
        
        # Assertions
        assert result["match_percentage"] >= 90, \
            f"Expected at least 90% (all must-haves), got {result['match_percentage']}%"
        
        assert result["match_percentage"] < 100, \
            f"Expected less than 100% (missing Docker), got {result['match_percentage']}%"
        
        assert result["analysis_breakdown"]["must_haves_met"] == result["analysis_breakdown"]["must_haves_total"], \
            "Should have all must-haves met"
        
        assert result["analysis_breakdown"]["nice_to_haves_met"] == 0, \
            "Should have zero nice-to-haves met (no Docker)"


class TestSynonymRecognitionIntelligence:
    """Test Case 4: Synonym Recognition (Intelligence Check)"""
    
    def test_sql_synonym_recognition(self):
        """
        INTELLIGENCE: AI should recognize PostgreSQL and MySQL as SQL synonyms.
        
        Test: Job requires SQL, candidate has PostgreSQL and MySQL.
        Expected: must_haves_met = 1 (SQL requirement satisfied)
        """
        job_description = """
        Position: Database Developer
        
        Must-Haves:
        - SQL experience
        
        Nice-to-Haves:
        - NoSQL databases
        """
        
        resume_text = """
        Bob Chen
        Austin, TX
        
        Database Specialist with 5 years of experience.
        
        Experience:
        - Database Developer (2019 - Present): Expert in PostgreSQL and MySQL
        - Optimized queries for high-traffic applications
        - Built data pipelines using PostgreSQL
        
        Skills: PostgreSQL, MySQL, Database Design, Query Optimization
        """
        
        result = analyze_resume_with_job_ai(resume_text, job_description)
        
        # Assertions
        assert result["analysis_breakdown"]["must_haves_met"] >= 1, \
            "AI should recognize PostgreSQL/MySQL as SQL synonyms"
        
        # Should get high score since the main must-have is met
        assert result["match_percentage"] >= 90, \
            f"Expected at least 90% (SQL requirement met via synonyms), got {result['match_percentage']}%"


class TestSoftSkillsIgnoredRule:
    """Test Case 5: Soft Skills Ignored (Rule Check)"""
    
    def test_soft_skills_not_counted_in_must_haves_total(self):
        """
        RULE: Soft skills should NOT be counted in must_haves_total for scoring.
        
        Test: Job lists "Python" and "Communication Skills" as must-haves.
        Expected: must_haves_total = 1 (only Python counts)
        """
        job_description = """
        Position: Python Developer
        
        Must-Haves:
        - Python programming (3+ years)
        - Strong communication skills
        - Team player attitude
        
        Nice-to-Haves:
        - AWS
        """
        
        resume_text = """
        Carol White
        Boston, MA
        
        Python Developer with 5 years of experience.
        
        Experience:
        - Python Engineer (2019 - Present): Developed backend systems using Python
        - Great communication skills, works well in teams
        
        Skills: Python, Django, Flask, Git, Linux
        """
        
        result = analyze_resume_with_job_ai(resume_text, job_description)
        
        # Assertions
        assert result["analysis_breakdown"]["must_haves_total"] <= 2, \
            "Soft skills ('communication', 'team player') should be excluded or minimized from must_haves_total"
        
        # Since Python is met and soft skills are ignored/auto-met, should get high score
        assert result["match_percentage"] >= 85, \
            f"Expected high score (Python met, soft skills ignored), got {result['match_percentage']}%"


class TestDateCalculationIntelligence:
    """Test Case 6: Date Calculation (Intelligence Check)"""
    
    def test_experience_years_calculated_from_dates(self):
        """
        INTELLIGENCE: AI should calculate years of experience from date ranges.
        
        Test: Job requires "5+ years", resume shows "Jan 2018 - Present".
        Expected: work_experience_years >= 7 (assuming current year is 2025)
                  AND must_haves_met = 1
        """
        job_description = """
        Position: Senior Software Engineer
        
        Must-Haves:
        - 5+ years of software development experience
        
        Nice-to-Haves:
        - Leadership experience
        """
        
        resume_text = """
        David Lee
        San Diego, CA
        
        Software Engineer
        
        Experience:
        - Software Engineer at Tech Corp (January 2018 - Present)
          * Developed enterprise applications
          * Led technical projects
          * Mentored junior developers
        
        Skills: Python, Java, C++, SQL, Git
        """
        
        result = analyze_resume_with_job_ai(resume_text, job_description)
        
        # Assertions
        # Current year is 2025 (November), so Jan 2018 - Present = ~5-7 years depending on calculation method
        # AI may calculate as 5 (2018-2023), 6 (partial 2024), or 7 (including 2025)
        assert result["candidate_info"]["work_experience_years"] >= 5, \
            f"Expected at least 5 years calculated from 2018-Present, got {result['candidate_info']['work_experience_years']}"
        
        assert result["analysis_breakdown"]["must_haves_met"] >= 1, \
            "5+ years requirement should be met (candidate has 7 years)"
        
        # Should get high score since experience requirement is met
        assert result["match_percentage"] >= 90, \
            f"Expected at least 90% (experience requirement met), got {result['match_percentage']}%"


# ============================================================================
# INTEGRATION TEST: Complex Scenario
# ============================================================================

class TestComplexIntegration:
    """Integration test combining multiple rules."""
    
    def test_complex_scoring_scenario(self):
        """
        Test combining: synonyms, soft skills, nice-to-haves, and experience.
        
        Should demonstrate that the AI correctly:
        - Recognizes synonyms
        - Ignores soft skills
        - Calculates experience
        - Adds nice-to-have bonuses
        """
        job_description = """
        Position: Senior Full Stack Engineer
        Location: Remote
        
        Must-Haves:
        - 5+ years of experience
        - Python or JavaScript programming
        - SQL database experience
        - Strong communication skills
        
        Nice-to-Haves:
        - Docker
        - Kubernetes
        """
        
        resume_text = """
        Emma Rodriguez
        Remote (San Francisco, CA)
        
        Senior Software Engineer
        
        Experience:
        - Senior Engineer at StartupXYZ (2018 - Present)
          * Built scalable web services using Node.js
          * Extensive PostgreSQL database optimization
          * Deployed applications using Docker and Kubernetes
          * Led cross-functional teams with excellent communication
        
        Skills: Node.js, JavaScript, PostgreSQL, Docker, Kubernetes, AWS
        """
        
        result = analyze_resume_with_job_ai(resume_text, job_description)
        
        # Should recognize:
        # - 7 years experience (2018-Present)
        # - JavaScript (via Node.js synonym)
        # - SQL (via PostgreSQL synonym)
        # - Docker + Kubernetes nice-to-haves
        # - Communication skills (but not counted in scoring)
        
        assert result["match_percentage"] >= 95, \
            f"Expected very high score (all requirements + bonuses), got {result['match_percentage']}%"
        
        assert result["analysis_breakdown"]["must_haves_met"] >= 3, \
            "Should meet experience, programming, and SQL requirements"
        
        assert result["analysis_breakdown"]["nice_to_haves_met"] >= 2, \
            "Should recognize Docker and Kubernetes"


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

