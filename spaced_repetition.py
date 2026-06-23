"""
Ebbinghaus forgetting curve spaced repetition scheduler (SM-2 variant).
Determines optimal next review time for each topic based on
student performance history.
"""
import os
import json
import math
from datetime import datetime, timedelta
from typing import Optional

# Ebbinghaus stability multipliers per performance level
PERFORMANCE_MULTIPLIERS = {
    "perfect":  2.5,
    "good":     2.0,
    "okay":     1.5,
    "poor":     1.0,
    "fail":     0.5,
}

DEFAULT_INTERVAL_DAYS = 1.0
SCHEDULE_FILE = "student_schedules.json"


def load_schedules() -> dict:
    if os.path.exists(SCHEDULE_FILE):
        try:
            with open(SCHEDULE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_schedules(schedules: dict):
    with open(SCHEDULE_FILE, "w") as f:
        json.dump(schedules, f, indent=2, default=str)


def next_review_interval(
    current_interval_days: float,
    performance: str,
    ease_factor: float = 2.5,
) -> tuple[float, float]:
    """
    SM-2 inspired algorithm.
    Returns (next_interval_days, updated_ease_factor).
    """
    perf_mult = PERFORMANCE_MULTIPLIERS.get(performance, 1.0)

    if performance == "fail":
        # Reset to day 1
        return DEFAULT_INTERVAL_DAYS, max(1.3, ease_factor - 0.2)

    next_interval = current_interval_days * ease_factor * perf_mult
    next_interval = max(1.0, min(next_interval, 365.0))  # cap at 1 year

    # Update ease factor
    if performance == "perfect":
        new_ease = min(3.0, ease_factor + 0.1)
    elif performance in ("good", "okay"):
        new_ease = ease_factor  # unchanged
    else:
        new_ease = max(1.3, ease_factor - 0.15)

    return round(next_interval, 2), round(new_ease, 3)


def get_due_topics_internal(topic_history: dict[str, dict]) -> list[dict]:
    """
    Given a dict of {topic: {last_reviewed, interval_days, ease_factor, accuracy}},
    return topics due for review today, sorted by urgency.
    """
    now = datetime.utcnow()
    due = []

    for topic, info in topic_history.items():
        last = datetime.fromisoformat(info["last_reviewed"])
        interval = info.get("interval_days", DEFAULT_INTERVAL_DAYS)
        due_date = last + timedelta(days=interval)
        days_overdue = (now - due_date).total_seconds() / 86400

        if days_overdue >= 0:
            due.append({
                "topic": topic,
                "due_date": due_date.isoformat(),
                "days_overdue": round(days_overdue, 2),
                "accuracy": info.get("accuracy", 0.0),
                "ease_factor": info.get("ease_factor", 2.5),
                "interval_days": interval,
                "priority": "HIGH" if days_overdue > 2 else "NORMAL",
            })

    # Sort: overdue + low accuracy first
    due.sort(key=lambda x: (-x["days_overdue"], x["accuracy"]))
    return due


def update_topic(
    topic_history: dict,
    topic: str,
    performance: str,
) -> dict:
    """Update a topic's schedule after a review session."""
    now = datetime.utcnow().isoformat()

    if topic not in topic_history:
        topic_history[topic] = {
            "last_reviewed": now,
            "interval_days": DEFAULT_INTERVAL_DAYS,
            "ease_factor": 2.5,
            "accuracy": 0.0,
            "reviews": 0,
        }

    info = topic_history[topic]
    new_interval, new_ease = next_review_interval(
        info["interval_days"], performance, info["ease_factor"]
    )

    # Running accuracy
    perf_score = {"perfect": 1.0, "good": 0.8, "okay": 0.6, "poor": 0.4, "fail": 0.0}
    reviews = info["reviews"] + 1
    old_acc = info["accuracy"]
    new_acc = (old_acc * (reviews - 1) + perf_score.get(performance, 0.5)) / reviews

    topic_history[topic] = {
        "last_reviewed": now,
        "interval_days": new_interval,
        "ease_factor": new_ease,
        "accuracy": round(new_acc, 4),
        "reviews": reviews,
    }
    return topic_history


# ── Server Endpoint Implementations ──────────────────────────────────────────

def get_due_topics(student_id: str) -> list[dict]:
    """Exposed for server endpoint: retrieves list of due topics."""
    schedules = load_schedules()
    student_history = schedules.get(student_id, {})
    due_internal = get_due_topics_internal(student_history)
    
    # Map to server response shape
    due_topics = []
    for item in due_internal:
        topic_id = item["topic"]
        info = student_history[topic_id]
        due_topics.append({
            "topic_id": topic_id,
            "topic_name": info.get("topic_name", topic_id),
            "review_count": info.get("reviews", 1),
            "last_score": info.get("accuracy", 0.0),
            "next_review": (datetime.fromisoformat(info["last_reviewed"]) + timedelta(days=info["interval_days"])).isoformat()
        })
    return due_topics


def record_review(student_id: str, topic_id: str, topic_name: str, score: float) -> dict:
    """Exposed for server endpoint: records a student's review result."""
    schedules = load_schedules()
    student_history = schedules.setdefault(student_id, {})

    # Map numerical score to performance string
    if score >= 0.9:
        performance = "perfect"
    elif score >= 0.7:
        performance = "good"
    elif score >= 0.5:
        performance = "okay"
    elif score >= 0.3:
        performance = "poor"
    else:
        performance = "fail"

    # Save additional metadata in the schedule
    if topic_id in student_history:
        student_history[topic_id]["topic_name"] = topic_name
    else:
        student_history[topic_id] = {
            "last_reviewed": datetime.utcnow().isoformat(),
            "interval_days": DEFAULT_INTERVAL_DAYS,
            "ease_factor": 2.5,
            "accuracy": 0.0,
            "reviews": 0,
            "topic_name": topic_name
        }

    update_topic(student_history, topic_id, performance)
    save_schedules(schedules)

    info = student_history[topic_id]
    next_date = (datetime.fromisoformat(info["last_reviewed"]) + timedelta(days=info["interval_days"])).isoformat()

    return {
        "topic_id": topic_id,
        "next_review_in_days": info["interval_days"],
        "next_review_date": next_date
    }


def get_student_analytics(student_id: str) -> dict:
    """Exposed for server endpoint: retrieves analytics of student progress."""
    schedules = load_schedules()
    student_history = schedules.get(student_id, {})

    if not student_history:
        return {"message": "No data found for this student."}

    topics = list(student_history.values())
    avg_score = sum(t.get("accuracy", 0.0) for t in topics) / len(topics)
    weak_areas = [t.get("topic_name", tid) for tid, t in student_history.items() if t.get("accuracy", 0.0) < 0.6]
    strong_areas = [t.get("topic_name", tid) for tid, t in student_history.items() if t.get("accuracy", 0.0) >= 0.8]

    return {
        "student_id": student_id,
        "total_topics_studied": len(topics),
        "average_score": round(avg_score, 2),
        "weak_areas": weak_areas,
        "strong_areas": strong_areas,
        "topics_due_today": len(get_due_topics(student_id))
    }
