# metrics_period.py
"""
Period metrics and derived KPIs for the Student Report app.

Exports (backward compatible):
- available_date_range_for_user(data, user_id, idx)
- period_stats(data, user_id, start_dt, end_dt)
- compute_trend(curr, prev)
- compute_focus_score(completion_pct, avg_session_mins)

Advanced Learning KPIs:
- accuracy_and_mastery(data, user_id, start_dt, end_dt)
- response_time_stats(data, user_id, start_dt, end_dt)
- engagement_consistency(data, user_id, start_dt, end_dt)
- independence_support(data, user_id, start_dt, end_dt)
- communication_social(data, user_id, start_dt, end_dt)
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from statistics import mean
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from joins import (
    build_indexes,
    topic_session_user_id,
    chapter_session_user_id,
    perf_user_id,
    perf_subject,
)
from utils_time import parse_ts


# -----------------------------
# Helpers
# -----------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _pct(n: float, d: float) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def _pick_first_ts(record: Dict[str, Any], keys: List[str]) -> Optional[datetime]:
    for k in keys:
        if k in record and record[k] not in (None, "", "Unknown"):
            dt = parse_ts(record[k])
            if dt:
                return dt
    return None


def _get_time_spent(r: Dict[str, Any]) -> float:
    v = r.get("total_time_spent", None)
    if v is None:
        v = r.get("time_spent", 0)
    try:
        return float(v or 0.0)
    except Exception:
        return 0.0


# -----------------------------
# Public: available date range
# -----------------------------
def available_date_range_for_user(
    data: Dict[str, Any], user_id: int, idx: Dict[str, Any]
) -> Tuple[Optional[datetime], Optional[datetime]]:
    dates: List[datetime] = []

    for r in data.get("daily_activity_log", []):
        if r.get("user_id") == user_id:
            dt = parse_ts(r.get("login_timestamp"))
            if dt:
                dates.append(dt)

    for r in data.get("lesson_session", []):
        if r.get("user_id") == user_id:
            dt = parse_ts(r.get("created_at"))
            if dt:
                dates.append(dt)

    for ts in data.get("topic_session", []):
        uid = topic_session_user_id(ts, idx)
        if uid == user_id:
            for k in ("started_at", "completed_at"):
                dt = parse_ts(ts.get(k))
                if dt:
                    dates.append(dt)

    for cs in data.get("chapter_session", []):
        uid = chapter_session_user_id(cs, idx)
        if uid == user_id:
            for k in ("started_at", "completed_at"):
                dt = parse_ts(cs.get(k))
                if dt:
                    dates.append(dt)

    for ap in data.get("activity_performance", []):
        uid = perf_user_id(ap, idx)
        if uid == user_id:
            dt = parse_ts(ap.get("submitted_at"))
            if dt:
                dates.append(dt)

    if not dates:
        return None, None
    return min(dates), max(dates)


# -----------------------------
# Public: period stats
# -----------------------------
def _filter_records_by_period(
    records: List[Dict[str, Any]], start_dt: datetime, end_dt: datetime, ts_keys: List[str]
) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        ts = _pick_first_ts(r, ts_keys)
        if ts is None:
            continue
        if start_dt <= ts <= end_dt:
            out.append(r)
    return out


def period_stats(
    data: Dict[str, Any], user_id: int, start_dt: datetime, end_dt: datetime
) -> Dict[str, Any]:
    ts_keys_sessions = ["started_at", "completed_at", "created_at", "timestamp", "date"]
    ts_keys_perf = ["submitted_at"]
    ts_keys_dailies = ["login_timestamp", "created_at", "timestamp", "date"]

    idx = build_indexes(data)

    topic_sessions_all = [t for t in data.get("topic_session", []) if topic_session_user_id(t, idx) == user_id]
    chapter_sessions_all = [c for c in data.get("chapter_session", []) if chapter_session_user_id(c, idx) == user_id]
    lesson_sessions_all = [l for l in data.get("lesson_session", []) if l.get("user_id") == user_id]
    perf_all = [ap for ap in data.get("activity_performance", []) if perf_user_id(ap, idx) == user_id]
    daily_logs_all = [d for d in data.get("daily_activity_log", []) if d.get("user_id") == user_id]

    topic_sessions = _filter_records_by_period(topic_sessions_all, start_dt, end_dt, ts_keys_sessions)
    chapter_sessions = _filter_records_by_period(chapter_sessions_all, start_dt, end_dt, ts_keys_sessions)
    lesson_sessions = _filter_records_by_period(lesson_sessions_all, start_dt, end_dt, ts_keys_sessions)
    perf_rows = _filter_records_by_period(perf_all, start_dt, end_dt, ts_keys_perf)
    daily_logs = _filter_records_by_period(daily_logs_all, start_dt, end_dt, ts_keys_dailies)

    had_ts = (
        any([topic_sessions, chapter_sessions, lesson_sessions, perf_rows, daily_logs])
        or any(_pick_first_ts(x, ts_keys_sessions) for x in (topic_sessions_all + chapter_sessions_all + lesson_sessions_all))
        or any(_pick_first_ts(x, ts_keys_perf) for x in perf_all)
        or any(_pick_first_ts(x, ts_keys_dailies) for x in daily_logs_all)
    )

    total_time = sum(_get_time_spent(r) for r in topic_sessions) + \
                 sum(_get_time_spent(r) for r in chapter_sessions) + \
                 sum(float(r.get("time_spent", 0) or 0) for r in daily_logs) + \
                 sum(_get_time_spent(r) for r in lesson_sessions)

    session_lengths = [*[_get_time_spent(r) for r in topic_sessions],
                       *[_get_time_spent(r) for r in chapter_sessions],
                       *[float(r.get("time_spent", 0) or 0) for r in daily_logs],
                       *[_get_time_spent(r) for r in lesson_sessions]]
    session_lengths = [s for s in session_lengths if s and s > 0]
    sessions_count = len(session_lengths)
    avg_session_len = mean(session_lengths) if session_lengths else 0.0

    completed = sum(1 for t in topic_sessions if float(t.get("completion_percent") or 0) >= 80)
    total_lessons = len(topic_sessions)
    completion_pct = _pct(completed, total_lessons)

    day_set = set()

    def _collect_dates(coll, keys):
        for r in coll:
            ts = _pick_first_ts(r, keys)
            if ts:
                day_set.add(ts.date())

    _collect_dates(topic_sessions, ts_keys_sessions)
    _collect_dates(chapter_sessions, ts_keys_sessions)
    _collect_dates(lesson_sessions, ts_keys_sessions)
    _collect_dates(daily_logs, ts_keys_dailies)
    _collect_dates(perf_rows, ts_keys_perf)
    active_days = len(day_set)

    scores = []
    for p in perf_rows:
        sv = p.get("score", None)
        if sv in (None, ""):
            sv = 100.0 if p.get("is_right") else 0.0
        scores.append(float(sv))
    avg_score = mean(scores) if scores else 0.0

    return {
        "had_ts": had_ts,
        "total_time_mins": round(float(total_time), 1),
        "sessions": sessions_count,
        "avg_session_mins": round(float(avg_session_len), 1),
        "lessons_done": completed,
        "lessons_total": total_lessons,
        "completion_pct": round(float(completion_pct), 1),
        "active_days": active_days,
        "avg_score": round(float(avg_score), 1),
    }


# -----------------------------
# Public: trends & focus score
# -----------------------------
def compute_trend(curr: float, prev: float) -> int:
    if prev <= 0:
        return 0
    return int(round(100 * (curr - prev) / prev))


def compute_focus_score(completion_pct: float, avg_session_mins: float) -> int:
    base = 50.0
    base += (completion_pct - 50.0) * 0.4
    base += (avg_session_mins - 10.0) * 1.2
    return int(_clamp(base, 0, 100))


# ============================================================
# Advanced Learning KPIs
# ============================================================
def accuracy_and_mastery(
    data: Dict[str, Any], user_id: int, start_dt: datetime, end_dt: datetime
) -> Dict[str, Any]:
    """
    Returns:
      {
        "overall": float,
        "subjects": {subject: avg%},
        "subjects_meta": [{"subject": str, "avg": float, "band": str, "attempts": int}]
      }
    """
    idx = build_indexes(data)
    rows: List[Dict[str, Any]] = []
    for ap in data.get("activity_performance", []):
        if perf_user_id(ap, idx) != user_id:
            continue
        dt = parse_ts(ap.get("submitted_at"))
        if not dt or not (start_dt <= dt <= end_dt):
            continue
        score = ap.get("score")
        if score in (None, ""):
            score = 100.0 if ap.get("is_right") else 0.0
        subj = perf_subject(ap, idx) or "Math"
        rows.append({"subject": subj, "score": float(score)})

    if not rows:
        return {"overall": 0.0, "subjects": {}, "subjects_meta": []}

    df = pd.DataFrame(rows)
    overall = float(df["score"].mean())

    def band(x: float) -> str:
        return "Mastered" if x >= 80 else ("Developing" if x >= 50 else "Emerging")

    grp = (
        df.groupby("subject")
          .agg(avg=("score", "mean"), attempts=("score", "size"))
          .reset_index()
    )
    grp["avg"] = grp["avg"].round(1)
    grp["band"] = grp["avg"].apply(band)

    subjects_dict = {r["subject"]: float(r["avg"]) for r in grp.to_dict(orient="records")}
    subjects_meta = grp[["subject", "avg", "band", "attempts"]].to_dict(orient="records")

    return {"overall": overall, "subjects": subjects_dict, "subjects_meta": subjects_meta}


def response_time_stats(
    data: Dict[str, Any], user_id: int, start_dt: datetime, end_dt: datetime
) -> Dict[str, Any]:
    """
    Primary: attempt-level time_spent per activity within the window.
    Fallback: if none available, estimate per-subject time from session durations.
    Returns dicts suitable for bar charts and SEN report bullets.
    """
    idx = build_indexes(data)

    # ---- Primary: attempt-level ----
    attempt_rows: List[Dict[str, Any]] = []
    for ap in data.get("activity_performance", []):
        if perf_user_id(ap, idx) != user_id:
            continue
        dt = parse_ts(ap.get("submitted_at"))
        if not dt or not (start_dt <= dt <= end_dt):
            continue
        t = ap.get("time_spent", 0) or 0
        try:
            t = float(t)
        except Exception:
            t = 0.0
        if t <= 0:
            continue
        subj = perf_subject(ap, idx) or "Math"
        attempt_rows.append({"subject": subj, "time_spent": t})

    if attempt_rows:
        df = pd.DataFrame(attempt_rows)
        arr = df["time_spent"].to_numpy(dtype=float)
        mean_t = float(np.mean(arr))
        med_t  = float(np.median(arr))
        p90_t  = float(np.quantile(arr, 0.9))

        per_subj = (
            df.groupby("subject")
              .agg(total_time=("time_spent", "sum"),
                   attempts=("time_spent", "size"),
                   avg_time=("time_spent", "mean"))
              .reset_index()
        )
        per_subj["total_time"] = per_subj["total_time"].round(1)
        per_subj["avg_time"]   = per_subj["avg_time"].round(1)

        return {
            "unit": "mins",  # change label if your time_spent is seconds
            "mean": round(mean_t, 1),
            "median": round(med_t, 1),
            "p90": round(p90_t, 1),
            "attempts": int(len(arr)),
            "per_subject": {r["subject"]: float(r["avg_time"]) for r in per_subj.to_dict(orient="records")},
            "per_subject_meta": per_subj[["subject", "avg_time", "total_time", "attempts"]].to_dict(orient="records"),
        }

    # ---- Fallback: session-level minutes per subject ----
    subj2mins: Dict[str, List[float]] = defaultdict(list)

    def add_session_time(coll, user_id_func, date_keys, subject_func):
        for r in coll:
            if user_id_func and user_id_func(r, idx) != user_id:
                continue
            if not user_id_func and r.get("user_id") != user_id:
                continue
            ts = None
            for k in date_keys:
                ts = parse_ts(r.get(k))
                if ts:
                    break
            if not ts or not (start_dt <= ts <= end_dt):
                continue
            v = r.get("total_time_spent") or r.get("time_spent") or 0
            try:
                mins = float(v)
            except Exception:
                mins = 0.0
            if mins <= 0:
                continue
            subj = subject_func(r) or "Unknown"
            subj2mins[subj].append(mins)

    def subj_from_topic_session(ts):
        en = idx["enrollment_by_id"].get(ts.get("enrollment_id"))
        if not en:
            return None
        t = idx["topics_by_id"].get(en.get("topic_id"))
        return t.get("subject") if t else None

    def subj_from_chapter_session(cs):
        ts = idx["topic_session_by_id"].get(cs.get("topic_session_id"))
        return subj_from_topic_session(ts) if ts else None

    add_session_time(
        data.get("topic_session", []),
        topic_session_user_id,
        ["started_at", "completed_at", "created_at"],
        subj_from_topic_session,
    )
    add_session_time(
        data.get("chapter_session", []),
        chapter_session_user_id,
        ["started_at", "completed_at"],
        subj_from_chapter_session,
    )
    add_session_time(
        data.get("lesson_session", []),
        None,
        ["created_at", "started_at", "completed_at"],
        lambda _: "Unknown",
    )

    if not subj2mins:
        return {
            "unit": "mins",
            "mean": 0.0, "median": 0.0, "p90": 0.0, "attempts": 0,
            "per_subject": {}, "per_subject_meta": []
        }

    rows = []
    for subj, mins_list in subj2mins.items():
        total_t = float(sum(mins_list))
        cnt = int(len(mins_list))
        avg_t = total_t / cnt if cnt else 0.0
        rows.append({"subject": subj, "avg_time": round(avg_t, 1), "total_time": round(total_t, 1), "attempts": cnt})

    df = pd.DataFrame(rows)
    arr = df["avg_time"].to_numpy(dtype=float)
    mean_t = float(np.mean(arr))
    med_t  = float(np.median(arr))
    p90_t  = float(np.quantile(arr, 0.9))

    return {
        "unit": "mins",
        "mean": round(mean_t, 1),
        "median": round(med_t, 1),
        "p90": round(p90_t, 1),
        "attempts": int(df["attempts"].sum()),
        "per_subject": {r["subject"]: float(r["avg_time"]) for r in rows},
        "per_subject_meta": rows,
    }


def engagement_consistency(
    data: Dict[str, Any], user_id: int, start_dt: datetime, end_dt: datetime
) -> Dict[str, Any]:
    stamps: List[datetime] = []

    def add_ts(ts):
        if ts:
            stamps.append(ts)

    for r in data.get("daily_activity_log", []):
        if r.get("user_id") == user_id:
            dt = parse_ts(r.get("login_timestamp"))
            if dt and start_dt <= dt <= end_dt:
                add_ts(dt)

    for coll, keys in [
        (data.get("lesson_session", []), ["created_at", "started_at", "completed_at"]),
        (data.get("topic_session", []), ["started_at", "completed_at", "created_at"]),
        (data.get("chapter_session", []), ["started_at", "completed_at"]),
    ]:
        for r in coll:
            for k in keys:
                dt = parse_ts(r.get(k))
                if dt and start_dt <= dt <= end_dt:
                    add_ts(dt)

    if not stamps:
        return {"active_days": 0, "sessions_total": 0, "streak": 0, "weeks": [], "heat": []}

    days = sorted(set([d.date() for d in stamps]))
    active_days = len(days)
    streak = best = 1
    for i in range(1, len(days)):
        if (days[i] - days[i - 1]).days == 1:
            streak += 1
            best = max(best, streak)
        else:
            streak = 1

    by_day = defaultdict(int)
    for dt in stamps:
        by_day[dt.date()] += 1

    start_w = start_dt.date() - timedelta(days=start_dt.weekday())
    end_d = end_dt.date()
    weeks = []
    cur = start_w
    while cur <= end_d:
        weeks.append(cur)
        cur += timedelta(days=7)

    heat = []
    for w in weeks:
        col = []
        for d in range(7):
            day = w + timedelta(days=d)
            col.append(by_day.get(day, 0))
        heat.append(col)

    return {
        "active_days": active_days,
        "sessions_total": len(stamps),
        "streak": best,
        "weeks": [w.isoformat() for w in weeks],
        "heat": heat,
    }


def independence_support(
    data: Dict[str, Any], user_id: int, start_dt: datetime, end_dt: datetime
) -> Dict[str, float]:
    idx = build_indexes(data)
    hints = 0
    attempts = 0
    retries = 0

    for ap in data.get("activity_performance", []):
        if perf_user_id(ap, idx) != user_id:
            continue
        dt = parse_ts(ap.get("submitted_at"))
        if not dt or not (start_dt <= dt <= end_dt):
            continue
        attempts += 1
        if ap.get("used_hint") in (True, 1, "true", "True"):
            hints += 1
        if ap.get("attempt_index", 0) not in (None, 0, "0"):
            retries += 1

    if attempts == 0:
        return {"hint_rate": 0.0, "retry_rate": 0.0, "independence": 0.0}

    hint_rate = 100.0 * hints / attempts
    retry_rate = 100.0 * retries / attempts
    independence = max(0.0, 100.0 - (0.6 * hint_rate + 0.4 * retry_rate))
    return {"hint_rate": round(hint_rate, 1), "retry_rate": round(retry_rate, 1), "independence": round(independence, 1)}


def communication_social(
    data: Dict[str, Any], user_id: int, start_dt: datetime, end_dt: datetime
) -> Dict[str, float]:
    msgs = 0
    if "messages" in data:
        for m in data["messages"]:
            if m.get("user_id") == user_id:
                dt = parse_ts(m.get("created_at") or m.get("timestamp"))
                if dt and start_dt <= dt <= end_dt:
                    msgs += 1

    # Proxy via personalisation usage
    try:
        from ui_personalisation import collect_personalisation_usage
        usage = collect_personalisation_usage(data, user_id, start_dt, end_dt)
        changes = sum(1 for d in usage.values() for _ in d.keys())
    except Exception:
        changes = 0

    return {"messages": float(msgs), "personalisation_changes": float(changes)}
