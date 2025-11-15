from app.processors.semantic_matcher import find_missing_must_haves

if __name__ == "__main__":
    must_haves = ["AWS certification", "backend systems", "Docker"]
    
    resume_skills = ["Python", "AWS", "Backend Development"]
    resume_certs = ["AWS Developer"]

    missing = find_missing_must_haves(must_haves, resume_skills, resume_certs)

    print("\nMust-Haves:", must_haves)
    print("Resume Skills:", resume_skills)
    print("Resume Certs:", resume_certs)
    print("Missing Must-Haves:", missing)
