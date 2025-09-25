# metrics_aggregate.py
from typing import Any, Dict, List, Tuple, Optional
from statistics import mean
from utils_time import get_time_spent, compute_age_from_dob, parse_ts
from joins import build_indexes, topic_session_user_id, chapter_session_user_id, perf_user_id, perf_subject

def aggregate_student(data: Dict[str, Any], user_id: int) -> Dict[str, Any]:
    idx = build_indexes(data)
    user = idx["users"].get(user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found.")

    total_points = 0.0
    total_time = 0.0
    all_session_lengths: List[float] = []

    ts_for_user = []
    for ts in data.get("topic_session", []):
        uid = topic_session_user_id(ts, idx)
        if uid == user_id:
            ts_for_user.append(ts)
            t = get_time_spent(ts); total_time += t
            if t > 0: all_session_lengths.append(t)
            total_points += float(ts.get("points_earned", 0) or 0)

    cs_for_user = []
    for cs in idx["chapter_sessions"]:
        uid = chapter_session_user_id(cs, idx)
        if uid == user_id:
            cs_for_user.append(cs)
            t = get_time_spent(cs); total_time += t
            if t > 0: all_session_lengths.append(t)
            total_points += float(cs.get("points_earned", 0) or 0)

    for d in idx["daily_logs"]:
        if d.get("user_id") == user_id:
            total_time += float(d.get("time_spent", 0) or 0)
            total_points += float(d.get("points_earned", 0) or 0)
            if d.get("time_spent"): all_session_lengths.append(float(d["time_spent"]))

    lesson_sessions = [l for l in idx["lesson_sessions"] if l.get("user_id") == user_id]
    for l in lesson_sessions:
        t = get_time_spent(l); total_time += t
        if t > 0: all_session_lengths.append(t)

    perf_rows = [ap for ap in idx["activity_perf"] if perf_user_id(ap, idx) == user_id]
    scores = [float(p.get("score")) if p.get("score") not in (None, "") else (100.0 if p.get("is_right") else 0.0)
              for p in perf_rows]
    avg_score = mean(scores) if scores else 0.0

    hints_used = [1.0 if (p.get("used_hint") in (True, 1, "true", "True")) else 0.0 for p in perf_rows]
    avg_hints_used = mean(hints_used) if hints_used else 0.0

    topics_completed = sum(1 for ts in ts_for_user if float(ts.get("completion_percent") or 0) >= 80)
    topics_total = len(ts_for_user)
    lesson_completion_rate = round(100 * topics_completed / topics_total, 1) if topics_total else 0.0

    ch_progress = [float(c.get("progress_percent") or 0) for c in cs_for_user]
    avg_chapter_progress = mean(ch_progress) if ch_progress else 0.0
    chapter_progress_summary = f"{len(cs_for_user)} chapters seen, average progress {avg_chapter_progress:.1f}%"
    avg_session_length = mean(all_session_lengths) if all_session_lengths else 0.0

    per_subject_series: Dict[str, List[Tuple[str, float]]] = {}
    for ap in perf_rows:
        subj = perf_subject(ap, idx) or "Unknown"
        dt = parse_ts(ap.get("submitted_at"))
        if dt:
            score_val = float(ap.get("score")) if ap.get("score") not in (None, "") else (100.0 if ap.get("is_right") else 0.0)
            per_subject_series.setdefault(subj, []).append((dt.date().isoformat(), score_val))

    dob = user.get("dob", "")
    aggregated = {
        "name": user.get("name", "Unknown Student"),
        "class_level": user.get("class_level", "Unknown"),
        "reading_level": user.get("reading_level", "Unknown"),
        "school_name": user.get("school_name", user.get("school", "Unknown")),
        "total_time": round(total_time, 1),
        "lessons_completed": topics_completed,
        "lesson_completion_rate": round(lesson_completion_rate, 1),
        "chapter_progress_summary": chapter_progress_summary,
        "avg_session_length": round(avg_session_length, 1),
        "avg_score": round(avg_score, 1),
        "total_points": round(total_points, 1),
        "avg_hints_used": round(avg_hints_used, 3),
        "dob": dob or "Unknown",
        "age_display": compute_age_from_dob(dob),
        "gender": user.get("gender", "Unknown"),
        "email": user.get("email", "Unknown"),
        "parental_email": user.get("parental_email", "Unknown"),
        "onboarding_complete": user.get("is_onboarding_complete", False),
        "created_at": user.get("created_at", "Unknown"),
        "updated_at": user.get("updated_at", "Unknown"),
        "avatar": user.get("avatarImg", None),
        "chapters_seen": len(cs_for_user),
        "avg_chapter_progress_val": round(avg_chapter_progress, 1),
        "subject_series": per_subject_series,
        "ts_for_user": ts_for_user,
        "cs_for_user": cs_for_user,
        "lesson_sessions": lesson_sessions,
        "perf_rows": perf_rows,
    }
    return aggregated
