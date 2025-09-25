# joins.py
from typing import Any, Dict, List, Optional
from utils_time import get_time_spent
def build_indexes(data: Dict[str, Any]):
    enrollment_by_id = {e["enrollment_id"]: e for e in data.get("enrollment", []) if "enrollment_id" in e}
    topics_by_id = {t["topic_id"]: t for t in data.get("topics", []) if "topic_id" in t}
    topic_session_by_id = {t["topic_session_id"]: t for t in data.get("topic_session", []) if "topic_session_id" in t}
    chapter_sessions = [c for c in data.get("chapter_session", []) if isinstance(c, dict)]
    lesson_sessions = [l for l in data.get("lesson_session", []) if isinstance(l, dict)]
    activity_perf = [a for a in data.get("activity_performance", []) if isinstance(a, dict)]
    daily_logs = [d for d in data.get("daily_activity_log", []) if isinstance(d, dict)]
    users = {u["user_id"]: u for u in data.get("user", []) if isinstance(u, dict) and "user_id" in u}
    return {
        "enrollment_by_id": enrollment_by_id,
        "topics_by_id": topics_by_id,
        "topic_session_by_id": topic_session_by_id,
        "chapter_sessions": chapter_sessions,
        "lesson_sessions": lesson_sessions,
        "activity_perf": activity_perf,
        "daily_logs": daily_logs,
        "users": users,
    }

def chapter_session_user_id(cs: Dict[str, Any], idx: Dict[str, Any]) -> Optional[int]:
    ts = idx["topic_session_by_id"].get(cs.get("topic_session_id"))
    if not ts:
        return None
    en = idx["enrollment_by_id"].get(ts.get("enrollment_id"))
    if not en:
        return None
    return en.get("user_id")

def topic_session_user_id(ts: Dict[str, Any], idx: Dict[str, Any]) -> Optional[int]:
    en = idx["enrollment_by_id"].get(ts.get("enrollment_id"))
    return en.get("user_id") if en else None

def perf_user_id(ap: Dict[str, Any], idx: Dict[str, Any]) -> Optional[int]:
    cs_id = ap.get("chapter_session_id")
    cs = next((c for c in idx["chapter_sessions"] if c.get("chapter_session_id") == cs_id), None)
    if not cs:
        return None
    return chapter_session_user_id(cs, idx)

def perf_subject(ap: Dict[str, Any], idx: Dict[str, Any]) -> Optional[str]:
    cs_id = ap.get("chapter_session_id")
    cs = next((c for c in idx["chapter_sessions"] if c.get("chapter_session_id") == cs_id), None)
    if not cs:
        return None
    ts = idx["topic_session_by_id"].get(cs.get("topic_session_id"))
    if not ts:
        return None
    en = idx["enrollment_by_id"].get(ts.get("enrollment_id"))
    if not en:
        return None
    topic = idx["topics_by_id"].get(en.get("topic_id"))
    return topic.get("subject") if topic else None
