"""
server.py — FastAPI backend for EduAgent exposing Q&A, MCQ generation, 
spaced repetition scheduling, and student analytics endpoints.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from agent import answer_question, generate_mcq
from spaced_repetition import get_due_topics, record_review, get_student_analytics

app = FastAPI(title="EduAgent AI Tutor API")


class QuestionRequest(BaseModel):
    student_id: str
    question: str


class MCQRequest(BaseModel):
    topic: str
    difficulty: str = "medium"


class ReviewRequest(BaseModel):
    student_id: str
    topic_id: str
    topic_name: str
    score: float


@app.post("/ask")
def ask(request: QuestionRequest):
    """RAG-grounded Q&A with dual-layer guardrails."""
    return answer_question(request.student_id, request.question)


@app.post("/mcq/generate")
def generate(request: MCQRequest):
    """Adaptive MCQ generation grounded in curriculum context."""
    return generate_mcq(request.topic, request.difficulty)


@app.get("/schedule/{student_id}")
def get_schedule(student_id: str):
    """Get topics due for spaced repetition review today."""
    return {"student_id": student_id, "due_topics": get_due_topics(student_id)}


@app.post("/schedule/review")
def submit_review(request: ReviewRequest):
    """Record a review result and schedule the next Ebbinghaus review."""
    return record_review(request.student_id, request.topic_id, request.topic_name, request.score)


@app.get("/analytics/{student_id}")
def analytics(student_id: str):
    """Student analytics: accuracy per topic, weak areas, learning velocity."""
    return get_student_analytics(student_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
