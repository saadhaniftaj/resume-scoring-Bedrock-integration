from app.processors.ai_matcher import analyze_resume_with_job

if __name__ == "__main__":
    job_description = """
We are looking for an experienced Computational Chemist to join the client's team and make an impact. You will help develop and apply molecular docking, molecular dynamics, and binding free energy calculation methods, and evaluate ML-optimized molecules from a computational chemistry perspective. You will also help incorporate pharmacological knowledge and intuition more effectively into the client's proprietary machine learning architecture.

Qualifications:
- Ph.D. in biophysics, computational chemistry, or a related discipline.
- 2+ years experience in small-molecule drug discovery.
- Experience with molecular modeling techniques (molecular docking, molecular dynamics, ABFE/RBFE).
- Proficiency with Schrodinger, GROMACS, Ambertools.
- Programming experience in Python; familiarity with RDKit, pandas, seaborn, SQL preferred.
- Intellectual curiosity and drive to excel.
    """

    resumes = [
        ("Perfect Match Chemist", """
Education:
Ph.D. in Computational Chemistry, University of Toronto, 2018

Technical Skills:
- Molecular Docking (Schrodinger, AutoDock)
- Molecular Dynamics (GROMACS, AMBER)
- Binding Free Energy Calculations (ABFE/RBFE)
- Python (RDKit, pandas, matplotlib, seaborn)
- SQL databases

Work Experience:
Computational Chemist, ABC Pharma (2018–Present)
- Designed molecular docking workflows using Schrodinger suite
- Performed free energy calculations (ABFE) to predict ligand affinity
- Integrated cheminformatics tools like RDKit for large-scale data analysis
- Collaborated with ML teams to optimize molecule libraries
        """),

        ("Missing Must-Haves Chemist", """
Education:
Ph.D. in Biophysics, McGill University, 2017

Technical Skills:
- Homology Modeling
- Protein Structure Prediction
- Basic Scripting (Bash, Python)

Work Experience:
Senior Research Scientist, DEF Biotech (2017–2023)
- Developed homology models for GPCR targets
- No direct experience with molecular dynamics or binding free energy methods
- Occasional Python scripting for data cleanup
        """),

        ("Python Only Scientist", """
Education:
M.Sc. in Bioinformatics, University of British Columbia, 2020

Technical Skills:
- Python (TensorFlow, PyTorch, pandas)
- Data Science (scikit-learn, SQL)
- Data Visualization (seaborn, matplotlib)

Work Experience:
Bioinformatics Analyst, GenomeTech Labs (2020–Present)
- Built deep learning models for protein classification
- No molecular docking or molecular dynamics experience
- Specialized in genomic and proteomic data analytics
        """),

        ("Wrong Domain Data Scientist", """
Education:
Ph.D. in Computer Science, University of Waterloo, 2016

Technical Skills:
- Machine Learning (XGBoost, LightGBM)
- Data Engineering (Airflow, Spark)
- Cloud Computing (AWS, GCP)

Work Experience:
Lead Data Scientist, XYZ Corp (2016–2023)
- Built predictive models for customer churn
- Designed scalable ETL pipelines
- No chemistry or drug discovery experience
        """),

        ("Good Chemistry, Bad Tech", """
Education:
Ph.D. in Medicinal Chemistry, University of British Columbia, 2015

Technical Skills:
- Molecular Docking (Schrodinger)
- Molecular Dynamics (AMBER)
- Structure-Based Drug Design

Work Experience:
Computational Chemist, BioHealth Solutions (2015–Present)
- Designed inhibitors using molecular docking and MD simulations
- No experience with Python, RDKit, or machine learning tools
- Strong chemistry background; limited technical programming skills
        """)
    ]

    # Update expected scores in test cases
    expected_scores = {
    "Perfect Match Chemist": (90, 100),
    "Missing Must-Haves Chemist": (20, 25), 
    "Python Only Scientist": (10, 25),
    "Wrong Domain Data Scientist": (0, 0),
    "Good Chemistry, Bad Tech": (60, 85)
}

for title, resume_text in resumes:
    print(f"\n=== Running Test: {title} ===")
    result = analyze_resume_with_job(resume_text, job_description)
    
    # Add validation
    min_score, max_score = expected_scores[title]
    assert min_score <= result["match_percentage"] <= max_score, (
        f"Test failed! Got score {result['match_percentage']} for:\n"
        f"Must-Haves: {result['must_have_results']}\n"
        f"Nice-to-Haves: {result['nice_have_results']}\n"
    )
