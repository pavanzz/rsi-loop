import re


def improve(target_code, feedback, llm):
    """Return a list of improved versions of `target_code`.

    feedback keys: task (str), score (float), failures (str), n_candidates (int)
    llm(prompt, temperature=0.0, max_tokens=1024) -> str, with a fixed call budget.
    """
    fence = "`" * 3
    prompt = (
        "You are improving a Python program.\n\n"
        f"TASK DESCRIPTION:\n{feedback['task']}\n\n"
        f"Current program (score {feedback['score']:.3f}):\n"
        f"{fence}python\n{target_code}\n{fence}\n\n"
        f"Feedback / failure examples:\n{feedback['failures'] or '(none)'}\n\n"
        "Write an improved version of the FULL program. Use only the Python standard "
        f"library. Reply with a single {fence}python code block."
    )
    candidates = []
    for _ in range(feedback["n_candidates"]):
        reply = llm(prompt, temperature=0.8, max_tokens=3000)
        blocks = re.findall(fence + r"python\n(.*?)" + fence, reply, re.S)
        if blocks:
            candidates.append(blocks[-1])
    return candidates
