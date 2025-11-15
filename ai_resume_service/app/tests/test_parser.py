from app.processors.local_parser import ResumeParser

def test_contractor_parsing():
    parser = ResumeParser()
    sample = """John Doe, Python Developer
    - 5 years AWS experience
    - Available for W2 contracts
    - BSc in Computer Science"""
    
    result = parser.parse(sample)
    assert "Python" in result["skills"]
    assert "W2" in result["contract_type"]
    assert result["education"] == "BSc"