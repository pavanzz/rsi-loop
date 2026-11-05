def solve(question, llm):
    prompt = (
        "Solve the following math word problem. Think step by step, then give the "
        "final numeric answer on the last line as 'Answer: <number>'.\n\n"
        f"Problem: {question}"
    )
    return llm(prompt)
