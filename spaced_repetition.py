"""
Ebbinghaus forgetting curve spaced repetition scheduler.
Determines optimal next review time for each topic based on
student performance history.
"""
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


def get_due_topics(topic_history: dict[str, dict]) -> list[dict]:
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
