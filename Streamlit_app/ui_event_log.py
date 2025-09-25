# ui_event_log.py
import streamlit as st
import pandas as pd
from constants import AVATAR_KEYS, FONT_KEYS, BACKGROUND_KEYS
from joins import build_indexes, topic_session_user_id, chapter_session_user_id, perf_user_id, perf_subject
from utils_time import get_time_spent

def _val(r, keys):
    for k in keys:
        if k in r and r[k] not in (None, ""):
            return r[k]
    for key in r.keys():
        lk = key.lower()
        for c in keys:
            if c.lower() in lk and r.get(key) not in (None, ""):
                return r.get(key)
    return "—"

def render_event_log_table(data, user_id: int):
    st.subheader("🧾 Per-event log (joined)")
    idx = build_indexes(data)
    rows = []

    for r in idx["daily_logs"]:
        if r.get("user_id") == user_id:
            rows.append({
                "event": "daily_login",
                "timestamp": r.get("login_timestamp"),
                "subject": "—",
                "score": "—",
                "points": r.get("points_earned", 0),
                "time_spent": r.get("time_spent", 0),
                "avatar": _val(r, AVATAR_KEYS),
                "font": _val(r, FONT_KEYS),
                "background": _val(r, BACKGROUND_KEYS),
                "chapter_session_id": "—",
                "topic_session_id": "—",
                "device": r.get("device_type", "—"),
            })

    for l in idx["lesson_sessions"]:
        if l.get("user_id") == user_id:
            rows.append({
                "event": "lesson_session",
                "timestamp": l.get("created_at"),
                "subject": "—",
                "score": "—",
                "points": l.get("points_earned", 0),
                "time_spent": get_time_spent(l),
                "avatar": _val(l, AVATAR_KEYS),
                "font": _val(l, FONT_KEYS),
                "background": _val(l, BACKGROUND_KEYS),
                "chapter_session_id": "—",
                "topic_session_id": "—",
                "device": l.get("device_type", "—"),
            })

    for ts in data.get("topic_session", []):
        uid = topic_session_user_id(ts, idx)
        if uid != user_id: continue
        en = idx["enrollment_by_id"].get(ts.get("enrollment_id"))
        subj = "—"
        if en:
            topic = idx["topics_by_id"].get(en.get("topic_id"))
            if topic: subj = topic.get("subject", "—")
        rows.append({
            "event": "topic_session",
            "timestamp": ts.get("started_at") or ts.get("completed_at"),
            "subject": subj,
            "score": "—",
            "points": ts.get("points_earned", 0),
            "time_spent": get_time_spent(ts),
            "avatar": _val(ts, AVATAR_KEYS),
            "font": _val(ts, FONT_KEYS),
            "background": _val(ts, BACKGROUND_KEYS),
            "chapter_session_id": "—",
            "topic_session_id": ts.get("topic_session_id"),
            "device": ts.get("device_type", "—"),
        })

    for cs in data.get("chapter_session", []):
        uid = chapter_session_user_id(cs, idx)
        if uid != user_id: continue
        ts = idx["topic_session_by_id"].get(cs.get("topic_session_id"))
        subj = "—"
        if ts:
            en = idx["enrollment_by_id"].get(ts.get("enrollment_id"))
            if en:
                topic = idx["topics_by_id"].get(en.get("topic_id"))
                if topic: subj = topic.get("subject", "—")
        rows.append({
            "event": "chapter_session",
            "timestamp": cs.get("started_at") or cs.get("completed_at"),
            "subject": subj,
            "score": "—",
            "points": cs.get("points_earned", 0),
            "time_spent": get_time_spent(cs),
            "avatar": _val(cs, AVATAR_KEYS),
            "font": _val(cs, FONT_KEYS),
            "background": _val(cs, BACKGROUND_KEYS),
            "chapter_session_id": cs.get("chapter_session_id"),
            "topic_session_id": cs.get("topic_session_id"),
            "device": "—",
        })

    for ap in data.get("activity_performance", []):
        uid = perf_user_id(ap, idx)
        if uid != user_id: continue
        subj = perf_subject(ap, idx) or "—"
        score_val = ap.get("score", None)
        if score_val in (None, ""): score_val = 100.0 if ap.get("is_right") else 0.0
        rows.append({
            "event": "activity_attempt",
            "timestamp": ap.get("submitted_at"),
            "subject": subj,
            "score": score_val,
            "points": ap.get("points_earned", 0),
            "time_spent": ap.get("time_spent", "—"),
            "avatar": _val(ap, AVATAR_KEYS),
            "font": _val(ap, FONT_KEYS),
            "background": _val(ap, BACKGROUND_KEYS),
            "chapter_session_id": ap.get("chapter_session_id"),
            "topic_session_id": "—",
            "device": "—",
        })

    if not rows:
        st.info("No events found for this user in the dataset.")
        return

    df = pd.DataFrame(rows).sort_values("timestamp")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download events (.csv)", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"user_{user_id}_events.csv", mime="text/csv")
