"""Thin, provider-agnostic LLM wrapper (any OpenAI-compatible endpoint).

Environment variables:
  RSI_MODEL     model name (required unless RSI_MOCK=1)
  RSI_API_KEY   API key (falls back to OPENAI_API_KEY)
  RSI_BASE_URL  optional base URL, e.g. https://openrouter.ai/api/v1 or a local server
  RSI_MOCK=1    use a fake LLM so the whole pipeline can be tested for free
"""
import os
import random
import re
import threading
import time


class LLM:
    def __init__(self):
        self.mock = os.environ.get("RSI_MOCK") == "1"
        self.model = os.environ.get("RSI_MODEL", "mock" if self.mock else "")
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self._lock = threading.Lock()
        if not self.mock:
            if not self.model:
                raise RuntimeError("Set RSI_MODEL (or RSI_MOCK=1 for a dry run).")
            from openai import OpenAI
            self.client = OpenAI(
                base_url=os.environ.get("RSI_BASE_URL") or None,
                api_key=os.environ.get("RSI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            )

    def __call__(self, prompt, temperature=0.0, max_tokens=1024):
        with self._lock:
            self.calls += 1
        if self.mock:
            return _mock_response(prompt)
        for attempt in range(6):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": str(prompt)}],
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                if r.usage:
                    with self._lock:
                        self.in_tokens += r.usage.prompt_tokens or 0
                        self.out_tokens += r.usage.completion_tokens or 0
                return r.choices[0].message.content or ""
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)


# ---------------------------------------------------------------- mock mode
def _mock_response(prompt):
    """Fake LLM: good enough to exercise every code path, not to produce science."""
    p = str(prompt)
    blocks = re.findall(r"```python\n(.*?)```", p, re.S)
    if blocks and ("improved version" in p.lower()):
        code = blocks[0]
        roll = random.random()
        if roll < 0.15:
            return "```python\ndef broken(:\n```"           # syntax error path
        if roll < 0.55 and "def solve" in code:
            # 'better' solver: asks twice and keeps the second answer
            new = code.rstrip() + (
                "\n\n_orig_solve = solve\n"
                "def solve(question, llm):\n"
                "    _orig_solve(question, llm)\n"
                "    return _orig_solve(question, llm)\n"
            )
            return f"```python\n{new}\n```"
        return f"```python\n# variant {random.randint(0, 10**6)}\n{code}\n```"
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", p.split("Problem:")[-1])]
    if len(nums) >= 2 and random.random() < 0.5:
        return f"Answer: {nums[0] + nums[1]:g}"
    return f"Answer: {random.choice(nums) if nums else 0:g}"
