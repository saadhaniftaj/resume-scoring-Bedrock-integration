from app.processors.deepseek_wrapper import DeepSeekLLM

if __name__ == "__main__":
    llm = DeepSeekLLM()
    response = llm.ask("Extract all skills and degree from this resume:\n\nJohn Smith has a BSc in Computer Science. He has worked with Python, React, and AWS.")
    print(response)