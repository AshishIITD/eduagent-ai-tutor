"""
Adaptive MCQ generator.
Adjusts difficulty based on student's running accuracy per topic.
"""
from pydantic import BaseModel
from typing import Literal
import json
import os


class MCQOption(BaseModel):
    key: str       # A, B, C, D
    text: str


class MCQ(BaseModel):
    question: str
    options: list[MCQOption]
    correct_key: str
    explanation: str
    difficulty: Literal["easy", "medium", "hard"]
    topic: str
    subject: str


class MCQResult(BaseModel):
    mcqs: list[MCQ]
    topic: str
    difficulty_used: str
    student_accuracy: float


def get_difficulty(accuracy: float) -> str:
    """
    Adaptive difficulty based on student's topic accuracy.
    accuracy < 0.5  → easy
    0.5 - 0.75      → medium
    > 0.75          → hard
    """
    if accuracy < 0.50:
        return "easy"
    elif accuracy < 0.75:
        return "medium"
    else:
        return "hard"


def build_mcq_prompt(topic: str, subject: str, difficulty: str, n: int = 3) -> str:
    difficulty_guidance = {
        "easy":   "straightforward recall questions with obvious wrong answers",
        "medium": "application-level questions requiring understanding",
        "hard":   "analysis and synthesis questions with plausible distractors",
    }
    return f"""Generate {n} multiple choice questions for exam preparation.

Topic: {topic}
Subject: {subject}
Difficulty: {difficulty} — {difficulty_guidance[difficulty]}
Exam context: Competitive entrance exam (UPSC/CAT/JEE/GATE level)

Return ONLY a valid JSON array. No preamble. No markdown. Format:
[
  {{
    "question": "Question text here?",
    "options": [
      {{"key": "A", "text": "Option A text"}},
      {{"key": "B", "text": "Option B text"}},
      {{"key": "C", "text": "Option C text"}},
      {{"key": "D", "text": "Option D text"}}
    ],
    "correct_key": "A",
    "explanation": "Brief explanation of why A is correct.",
    "difficulty": "{difficulty}",
    "topic": "{topic}",
    "subject": "{subject}"
  }}
]"""


def parse_mcq_response(raw: str, topic: str, subject: str, difficulty: str) -> list[MCQ]:
    """Parse LLM JSON response into MCQ objects."""
    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    data = json.loads(raw)
    mcqs = []
    for item in data:
        # Validate correct_key is one of the options
        option_keys = [o["key"] for o in item["options"]]
        if item["correct_key"] not in option_keys:
            item["correct_key"] = option_keys[0]

        mcqs.append(MCQ(
            question=item["question"],
            options=[MCQOption(**o) for o in item["options"]],
            correct_key=item["correct_key"],
            explanation=item["explanation"],
            difficulty=item.get("difficulty", difficulty),
            topic=item.get("topic", topic),
            subject=item.get("subject", subject),
        ))
    return mcqs


def score_attempt(mcq: MCQ, student_answer: str) -> dict:
    """Score a student's answer to an MCQ."""
    correct = student_answer.strip().upper() == mcq.correct_key.upper()
    return {
        "question": mcq.question,
        "student_answer": student_answer,
        "correct_answer": mcq.correct_key,
        "is_correct": correct,
        "explanation": mcq.explanation,
        "performance": "perfect" if correct else "fail",
    }
