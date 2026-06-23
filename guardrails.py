"""
Dual-layer guardrail system for EduAgent.
Layer 1: Rule-based input filter  → blocks off-topic / harmful queries
Layer 2: LLM-based output checker → detects hallucination vs grounded answer
"""
import re
from typing import Literal

# ── Layer 1: Input filter ────────────────────────────────────────────────────

ALLOWED_DOMAINS = [
    "math", "mathematics", "algebra", "geometry", "arithmetic", "calculus",
    "physics", "chemistry", "biology", "science", "history", "geography",
    "english", "grammar", "comprehension", "vocabulary",
    "general knowledge", "current affairs", "reasoning", "aptitude",
    "upsc", "cat", "gate", "jee", "neet", "sainik", "ias", "ips",
    "exam", "study", "learn", "explain", "question", "answer", "concept",
    "formula", "theorem", "definition", "solve", "calculate", "describe",
    "what is", "how does", "why is", "who invented", "when did",
]

BLOCKED_PATTERNS = [
    r"\b(movie|film|actor|actress|bollywood|cricket|ipl|football|music|song|lyrics)\b",
    r"\b(hack|exploit|weapon|drug|illegal|cheat sheet)\b",
    r"\b(girlfriend|boyfriend|dating|romance|sex)\b",
    r"(write my assignment|do my homework for me|give me all answers)",
]


def check_input(query: str) -> dict:
    """
    Returns: {allowed: bool, reason: str, category: str}
    """
    q_lower = query.lower().strip()

    # Block pattern check
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return {
                "allowed": False,
                "reason": "Query is off-topic or violates content policy.",
                "category": "BLOCKED"
            }

    # Length check
    if len(q_lower) < 3:
        return {"allowed": False, "reason": "Query too short.", "category": "BLOCKED"}

    # Domain allowlist check (soft — if any keyword matches, allow)
    for domain_kw in ALLOWED_DOMAINS:
        if domain_kw in q_lower:
            return {"allowed": True, "reason": "Topic matches allowed educational domain.", "category": "ALLOWED"}

    # If no domain keyword matched but query looks like a question, allow with caution
    question_indicators = ["?", "what", "how", "why", "when", "who", "explain", "define", "describe"]
    if any(qi in q_lower for qi in question_indicators):
        return {"allowed": True, "reason": "General educational question.", "category": "ALLOWED"}

    # Default: block ambiguous non-educational queries
    return {
        "allowed": False,
        "reason": "Query does not appear to be related to exam preparation topics.",
        "category": "BLOCKED"
    }


# ── Layer 2: Output hallucination detector ────────────────────────────────────

def check_output_grounding(answer: str, context_docs: list[str]) -> dict:
    """
    Rule-based hallucination detection.
    Checks if key claims in the answer appear in the retrieved context.
    Returns: {grounded: bool, confidence: float, verdict: str}
    """
    if not context_docs:
        return {"grounded": False, "confidence": 0.0, "verdict": "HALLUCINATED",
                "reason": "No context retrieved to ground the answer."}

    if not answer or len(answer.strip()) < 10:
        return {"grounded": False, "confidence": 0.0, "verdict": "HALLUCINATED",
                "reason": "Answer is empty or too short."}

    combined_context = " ".join(context_docs).lower()
    answer_lower = answer.lower()

    # Extract key noun phrases from answer (simple: 3+ char non-stop words)
    stop_words = {"the","a","an","is","are","was","were","be","been","being",
                  "have","has","had","do","does","did","will","would","could",
                  "should","may","might","must","to","of","in","on","at","for",
                  "with","by","from","as","into","through","during","this","that"}

    answer_words = [w.strip(".,!?;:\"'()") for w in answer_lower.split()
                    if len(w) > 3 and w not in stop_words]

    if not answer_words:
        return {"grounded": True, "confidence": 0.5, "verdict": "UNCERTAIN",
                "reason": "Could not extract key terms to verify."}

    # Count how many answer keywords appear in context
    matched = sum(1 for w in answer_words if w in combined_context)
    coverage = matched / len(answer_words)

    if coverage >= 0.55:
        verdict = "GROUNDED"
        grounded = True
    elif coverage >= 0.30:
        verdict = "PARTIALLY_GROUNDED"
        grounded = True
    else:
        verdict = "HALLUCINATED"
        grounded = False

    return {
        "grounded": grounded,
        "confidence": round(coverage, 4),
        "verdict": verdict,
        "reason": f"{matched}/{len(answer_words)} key terms found in retrieved context.",
        "keyword_coverage": f"{coverage:.1%}"
    }


def input_guardrail(query: str) -> dict:
    """Backward compatibility wrapper for check_input."""
    return check_input(query)


def output_guardrail(question: str, answer: str, context: str) -> dict:
    """Backward compatibility wrapper for check_output_grounding."""
    res = check_output_grounding(answer, [context])
    return {
        "safe": res["grounded"],
        "verdict": res["verdict"],
        "reason": res["reason"]
    }

