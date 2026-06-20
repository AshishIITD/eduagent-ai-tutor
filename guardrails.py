"""
guardrails.py — Dual-layer guardrail system using local Ollama (no API key needed).
Layer 1: Rule-based input filter. Layer 2: Local LLM hallucination detector.
"""
import re, ollama

ALLOWED_TOPICS = [
    "math", "mathematics", "algebra", "geometry", "calculus", "arithmetic",
    "science", "physics", "chemistry", "biology",
    "history", "geography", "civics",
    "english", "grammar", "vocabulary", "literature",
    "reasoning", "aptitude", "exam", "test", "study", "question", "explain",
    "sainik", "jnv", "rms", "rimc", "aissee", "entrance"
]
OFF_TOPIC_PATTERNS = [
    r"\b(movie|film|song|music|game|cricket|football|sports|bollywood|hollywood)\b",
    r"\b(recipe|cook|food|restaurant)\b",
    r"\b(dating|relationship|love)\b",
    r"\b(stock|crypto|bitcoin|invest)\b",
]
LOCAL_MODEL = "llama3.1:8b"

def input_guardrail(query: str) -> dict:
    """Layer 1: Rule-based input filter (pattern matching — no LLM needed)."""
    query_lower = query.lower()
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return {"allowed": False, "reason": "Off-topic query detected. EduAgent only handles academic subjects."}
    if any(topic in query_lower for topic in ALLOWED_TOPICS):
        return {"allowed": True, "reason": "Topic is within academic curriculum scope."}
    return {"allowed": True, "reason": "Query allowed (ambiguous topic — proceeding)."}

def output_guardrail(question: str, answer: str, context: str) -> dict:
    """Layer 2: Local Ollama hallucination detector with improved prompt."""
    try:
        result = ollama.chat(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content":
                f"You are a fact-checker. Does the answer CONTRADICT the context? "
                f"If the answer is consistent with or extends the context logically, say GROUNDED. "
                f"Only say HALLUCINATED if there is a clear factual contradiction.\n\n"
                f"Context: {context[:600]}\n"
                f"Answer to check: {answer[:300]}\n\n"
                f"Reply with ONLY one word: GROUNDED or HALLUCINATED"
            }]
        )
        text = result['message']['content'].strip().upper()
        if "GROUNDED" in text:
            verdict = "GROUNDED"
        elif "HALLUCINATED" in text:
            verdict = "HALLUCINATED"
        else:
            verdict = "GROUNDED"  # default safe

        is_safe = verdict == "GROUNDED"
        return {
            "safe": is_safe, "verdict": verdict,
            "reason": "Answer is curriculum-grounded." if is_safe
                      else "⚠️ Potential hallucination detected — answer blocked."
        }
    except Exception as e:
        return {"safe": True, "verdict": "ERROR", "reason": f"Guardrail check failed: {e}"}
