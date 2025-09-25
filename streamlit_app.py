# streamlit_app.py
import streamlit as st
import re
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, date, timedelta
import pandas as pd
import altair as alt
import plotly.graph_objects as go  # <-- Plotly for full pies

# ---------- Defaults ----------
DEFAULT_JSON_PATH = "fake_data.json"

st.set_page_config(page_title="Student Report Generator", layout="wide")


# ---------- Utilities ----------
def parse_ts(ts: Any) -> Optional[datetime]:
    """Parse many common timestamp formats. Returns None if not parseable."""
    if ts is None:
        return None
    if isinstance(ts, (datetime, pd.Timestamp)):
        try:
            return pd.to_datetime(ts).to_pydatetime()
        except Exception:
            return None
    s = str(ts).strip()
    fmts = [
        "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d"
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except Exception:
            continue
    try:
        dt = pd.to_datetime(s, errors="coerce")
        return None if pd.isna(dt) else dt.to_pydatetime()
    except Exception:
        return None


def pick_first_ts(record: Dict[str, Any], keys: List[str]) -> Optional[datetime]:
    for k in keys:
        if k in record and record[k] not in (None, "", "Unknown"):
            dt = parse_ts(record[k])
            if dt:
                return dt
    return None


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def pct(n: float, d: float) -> float:
    return round(100 * n / d, 1) if d else 0.0


def get_time_spent(r: Dict[str, Any]) -> float:
    """Use either total_time_spent or time_spent if present (minutes)."""
    v = r.get("total_time_spent", None)
    if v is None:
        v = r.get("time_spent", 0)
    try:
        return float(v or 0.0)
    except Exception:
        return 0.0


def compute_age_from_dob(dob_str: str) -> str:
    """Return years as a string, or '—' if not available/invalid."""
    if not dob_str:
        return "—"
    dob = None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            dob = datetime.strptime(dob_str, fmt).date()
            break
        except Exception:
            pass
    if dob is None:
        try:
            dob = datetime.fromisoformat(dob_str).date()
        except Exception:
            return "—"
    today = date.today()
    years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return str(years)


# ---------- Loading & normalization (works for fake_data.json and student_fake_data.json) ----------
@st.cache_data(show_spinner=False)
def load_json_raw(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_union(raw) -> Dict[str, Any]:
    """
    Accepts either:
      - A flat dict (like student_fake_data.json)
      - An array of per-user bundles (like fake_data.json) with keys including 'Topics'
    Returns a single dict of lists with unified key 'topics'.
    """
    if isinstance(raw, dict):
        data = dict(raw)
        if "topics" not in data and "Topics" in data:
            data["topics"] = data.get("Topics", [])
        for k in ["user", "enrollment", "daily_activity_log", "topic_session",
                  "chapter_session", "activity_performance", "lesson_session", "topics"]:
            data.setdefault(k, [])
        return data

    buckets: Dict[str, List[Any]] = {}
    for bundle in raw:
        if not isinstance(bundle, dict):
            continue
        for k, v in bundle.items():
            key = "topics" if k == "Topics" else k
            if isinstance(v, list):
                buckets.setdefault(key, []).extend(v)
            else:
                buckets.setdefault(key, []).append(v)
    for k in ["user", "enrollment", "daily_activity_log", "topic_session",
              "chapter_session", "activity_performance", "lesson_session", "topics"]:
        buckets.setdefault(k, [])
    return buckets


@st.cache_data(show_spinner=False)
def load_json(path: str) -> Dict[str, Any]:
    raw = load_json_raw(path)
    return normalize_union(raw)


def extract_user_id_and_audience(query: str):
    m = re.search(r'user[_\s]?id\s*[:=]?\s*(\d+)', query, flags=re.IGNORECASE)
    user_id = int(m.group(1)) if m else None
    audience = "parent"
    if re.search(r'\bteacher\b', query, flags=re.IGNORECASE):
        audience = "teacher"
    return user_id, audience


# ---------- Joining helpers ----------
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


def available_date_range_for_user(data: Dict[str, Any], user_id: int, idx: Dict[str, Any]) -> Tuple[Optional[datetime], Optional[datetime]]:
    dates: List[datetime] = []
    # daily_activity_log
    for r in idx["daily_logs"]:
        if r.get("user_id") == user_id:
            dt = parse_ts(r.get("login_timestamp"))
            if dt:
                dates.append(dt)
    # lesson_session
    for r in idx["lesson_sessions"]:
        if r.get("user_id") == user_id:
            dt = parse_ts(r.get("created_at"))
            if dt:
                dates.append(dt)
    # topic_session
    for ts in data.get("topic_session", []):
        uid = topic_session_user_id(ts, idx)
        if uid == user_id:
            for k in ("started_at", "completed_at"):
                dt = parse_ts(ts.get(k))
                if dt:
                    dates.append(dt)
    # chapter_session
    for cs in idx["chapter_sessions"]:
        uid = chapter_session_user_id(cs, idx)
        if uid == user_id:
            for k in ("started_at", "completed_at"):
                dt = parse_ts(cs.get(k))
                if dt:
                    dates.append(dt)
    # activity_performance
    for ap in idx["activity_perf"]:
        uid = perf_user_id(ap, idx)
        if uid == user_id:
            dt = parse_ts(ap.get("submitted_at"))
            if dt:
                dates.append(dt)

    if not dates:
        return None, None
    return min(dates), max(dates)


# ---------- Aggregation from raw events ----------
def aggregate_student(data: Dict[str, Any], user_id: int) -> Dict[str, Any]:
    idx = build_indexes(data)

    user = idx["users"].get(user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found.")

    total_points = 0.0
    total_time = 0.0
    all_session_lengths: List[float] = []

    # Topic sessions
    ts_for_user = []
    for ts in data.get("topic_session", []):
        uid = topic_session_user_id(ts, idx)
        if uid == user_id:
            ts_for_user.append(ts)
            t = get_time_spent(ts)
            total_time += t
            if t > 0:
                all_session_lengths.append(t)
            total_points += float(ts.get("points_earned", 0) or 0)

    # Chapter sessions
    cs_for_user = []
    for cs in idx["chapter_sessions"]:
        uid = chapter_session_user_id(cs, idx)
        if uid == user_id:
            cs_for_user.append(cs)
            t = get_time_spent(cs)
            total_time += t
            if t > 0:
                all_session_lengths.append(t)
            total_points += float(cs.get("points_earned", 0) or 0)

    # Daily logs
    for d in idx["daily_logs"]:
        if d.get("user_id") == user_id:
            total_time += float(d.get("time_spent", 0) or 0)
            total_points += float(d.get("points_earned", 0) or 0)
            if d.get("time_spent"):
                all_session_lengths.append(float(d["time_spent"]))

    # Lesson sessions
    lesson_sessions = [l for l in idx["lesson_sessions"] if l.get("user_id") == user_id]
    for l in lesson_sessions:
        t = get_time_spent(l)
        total_time += t
        if t > 0:
            all_session_lengths.append(t)

    # Performance rows
    perf_rows = [ap for ap in idx["activity_perf"] if perf_user_id(ap, idx) == user_id]
    scores = [float(p.get("score")) if p.get("score") not in (None, "") else (100.0 if p.get("is_right") else 0.0)
              for p in perf_rows]
    avg_score = mean(scores) if scores else 0.0

    hints_used = [1.0 if (p.get("used_hint") in (True, 1, "true", "True")) else 0.0 for p in perf_rows]
    avg_hints_used = mean(hints_used) if hints_used else 0.0

    # Completion via topic_session.completion_percent >= 80
    topics_completed = sum(1 for ts in ts_for_user if float(ts.get("completion_percent") or 0) >= 80)
    topics_total = len(ts_for_user)
    lesson_completion_rate = pct(topics_completed, topics_total)

    # Chapter progress
    ch_progress = [float(c.get("progress_percent") or 0) for c in cs_for_user]
    avg_chapter_progress = mean(ch_progress) if ch_progress else 0.0
    chapter_progress_summary = f"{len(cs_for_user)} chapters seen, average progress {avg_chapter_progress:.1f}%"

    avg_session_length = mean(all_session_lengths) if all_session_lengths else 0.0

    # Subject growth series
    per_subject_series: Dict[str, List[Tuple[str, float]]] = {}
    for ap in perf_rows:
        subj = perf_subject(ap, idx) or "Unknown"
        dt = parse_ts(ap.get("submitted_at"))
        if dt:
            score_val = float(ap.get("score")) if ap.get("score") not in (None, "") else (100.0 if ap.get("is_right") else 0.0)
            per_subject_series.setdefault(subj, []).append((dt.date().isoformat(), score_val))

    # User basics
    dob = user.get("dob", "")
    age_display = compute_age_from_dob(dob)
    gender = user.get("gender", "Unknown")
    email = user.get("email", "Unknown")
    parental_email = user.get("parental_email", "Unknown")
    avatar = user.get("avatarImg", None)

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
        "age_display": age_display,
        "gender": gender,
        "email": email,
        "parental_email": parental_email,
        "onboarding_complete": user.get("is_onboarding_complete", False),
        "created_at": user.get("created_at", "Unknown"),
        "updated_at": user.get("updated_at", "Unknown"),
        "avatar": avatar,

        # Extra numeric fields for all-time visuals
        "chapters_seen": len(cs_for_user),
        "avg_chapter_progress_val": round(avg_chapter_progress, 1),

        # Per-subject time series derived from events
        "subject_series": per_subject_series,

        # For tables/charts
        "ts_for_user": ts_for_user,
        "cs_for_user": cs_for_user,
        "lesson_sessions": lesson_sessions,
        "perf_rows": perf_rows,
    }
    return aggregated


# ---------- Period-based metrics ----------
def filter_records_by_period(records: List[Dict[str, Any]], start_dt: datetime, end_dt: datetime,
                             ts_keys: List[str]) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        ts = pick_first_ts(r, ts_keys)
        if ts is None:
            continue
        if start_dt <= ts <= end_dt:
            out.append(r)
    return out


def period_stats(data: Dict[str, Any], user_id: int,
                 start_dt: datetime, end_dt: datetime) -> Dict[str, Any]:
    ts_keys_sessions = ["started_at", "completed_at", "created_at", "timestamp", "date"]
    ts_keys_perf = ["submitted_at"]
    ts_keys_dailies = ["login_timestamp", "created_at", "timestamp", "date"]

    idx = build_indexes(data)

    topic_sessions_all = [t for t in data.get("topic_session", []) if topic_session_user_id(t, idx) == user_id]
    chapter_sessions_all = [c for c in data.get("chapter_session", []) if chapter_session_user_id(c, idx) == user_id]
    lesson_sessions_all = [l for l in data.get("lesson_session", []) if l.get("user_id") == user_id]
    perf_all = [ap for ap in data.get("activity_performance", []) if perf_user_id(ap, idx) == user_id]
    daily_logs_all = [d for d in data.get("daily_activity_log", []) if d.get("user_id") == user_id]

    topic_sessions = filter_records_by_period(topic_sessions_all, start_dt, end_dt, ts_keys_sessions)
    chapter_sessions = filter_records_by_period(chapter_sessions_all, start_dt, end_dt, ts_keys_sessions)
    lesson_sessions = filter_records_by_period(lesson_sessions_all, start_dt, end_dt, ts_keys_sessions)
    perf_rows = filter_records_by_period(perf_all, start_dt, end_dt, ts_keys_perf)
    daily_logs = filter_records_by_period(daily_logs_all, start_dt, end_dt, ts_keys_dailies)

    had_ts = any([topic_sessions, chapter_sessions, lesson_sessions, perf_rows, daily_logs]) or \
        any(pick_first_ts(x, ts_keys_sessions) for x in (topic_sessions_all + chapter_sessions_all + lesson_sessions_all)) or \
        any(pick_first_ts(x, ts_keys_perf) for x in perf_all) or \
        any(pick_first_ts(x, ts_keys_dailies) for x in daily_logs_all)

    total_time = sum(get_time_spent(r) for r in topic_sessions) + \
                 sum(get_time_spent(r) for r in chapter_sessions) + \
                 sum(float(r.get("time_spent", 0) or 0) for r in daily_logs) + \
                 sum(get_time_spent(r) for r in lesson_sessions)

    session_lengths = [get_time_spent(r) for r in topic_sessions] + \
                      [get_time_spent(r) for r in chapter_sessions] + \
                      [float(r.get("time_spent", 0) or 0) for r in daily_logs] + \
                      [get_time_spent(r) for r in lesson_sessions]
    session_lengths = [s for s in session_lengths if s and s > 0]
    sessions_count = len(session_lengths)
    avg_session_len = mean(session_lengths) if session_lengths else 0.0

    completed = sum(1 for t in topic_sessions if float(t.get("completion_percent") or 0) >= 80)
    total_lessons = len(topic_sessions)
    completion_pct = pct(completed, total_lessons)

    day_set = set()
    def collect_dates(coll, keys):
        for r in coll:
            ts = pick_first_ts(r, keys)
            if ts:
                day_set.add(ts.date())
    collect_dates(topic_sessions, ts_keys_sessions)
    collect_dates(chapter_sessions, ts_keys_sessions)
    collect_dates(lesson_sessions, ts_keys_sessions)
    collect_dates(daily_logs, ts_keys_dailies)
    collect_dates(perf_rows, ts_keys_perf)
    active_days = len(day_set)

    scores = [float(p.get("score")) if p.get("score") not in (None, "") else (100.0 if p.get("is_right") else 0.0)
              for p in perf_rows]
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
        "avg_score": round(float(avg_score), 1)
    }


def compute_trend(curr: float, prev: float) -> int:
    if prev <= 0:
        return 0
    return int(round(100 * (curr - prev) / prev))


def compute_focus_score(completion_pct: float, avg_session_mins: float) -> int:
    base = 50.0
    base += (completion_pct - 50.0) * 0.4
    base += (avg_session_mins - 10.0) * 1.2
    return int(clamp(base, 0, 100))


# ---------- Report builder ----------
def arrow(delta: int) -> str:
    return "↑" if delta >= 0 else "↓"


def build_report(d: Dict[str, Any]) -> str:
    def pp01(x):
        try:
            return f"{round(float(x)*100)}%"
        except Exception:
            return "0%"

    rep = []
    rep.append("Student Learning Report (SEN)\n")
    rep.append(f"Student: {d['student'].get('name','')}")
    rep.append(f"Student ID: {d['student'].get('id','')}")
    rep.append(f"Class / Year: {d['student'].get('class','')} / {d['student'].get('year','')}")
    rep.append(f"Reporting Period: {d['period']['start']} – {d['period']['end']}")
    rep.append(f"Prepared for: {d.get('prepared_for','')}")
    rep.append(f"Generated on: {d['period']['generated_on']}")
    rep.append("Data Sources: activity_performance, chapter_session, topic_session, lesson_session, daily_activity_log, topics, enrollment\n")

    focus_delta = d["focus"].get("focus_score_delta", 0)
    comp_pct = d["usage"].get("completion_pct", 0)
    if focus_delta >= 5 and comp_pct >= 70:
        exec_summary = "Engagement and comprehension are improving; routine and supports appear to help."
    elif comp_pct < 40 or d["focus"]["focus_score"] < d["focus"].get("class_median", d["focus"]["focus_score"]):
        exec_summary = "Engagement, focus, and skill growth have declined compared to last period. Immediate teacher support is advised."
    else:
        exec_summary = "Overall progress is steady with moderate gains; continue current routine and supports."
    rep.append("1) Executive Summary")
    rep.append(f"Summary: {exec_summary}")
    rep.append(f"Focus score: {d['focus']['focus_score']} ({arrow(focus_delta)} {abs(focus_delta)} from last period)")
    rep.append(f"Completion rate: {d['usage']['lessons_done']}/{d['usage']['lessons_total']} ({d['usage']['completion_pct']}%)")
    rep.append(f"Time-on-task: {d['usage']['total_time_mins']} mins total ({d['usage'].get('trend_vs_prev_pct',0)}% vs last period)\n")

    rep.append("2) SEN Profile & Accommodations")
    rep.append("Summary: Derived metrics only (no pre-set accommodations in source).")
    rep.append(f"Primary Needs: —")
    rep.append(f"Accommodations: —")
    rep.append(f"Effectiveness: TTS ON → 0% vs OFF → 0% (0pp)")
    rep.append(f"Stability: Font size changed 0× this period\n")

    eng_summary = "Strong participation and lesson completion." if comp_pct >= 70 else \
                  "Very low engagement, with limited active days and short sessions." if comp_pct < 40 else \
                  "Moderate engagement; room for higher completion."
    rep.append("3) Engagement & Usage")
    rep.append(f"Summary: {eng_summary}")
    rep.append(f"Active Days: {d['usage'].get('active_days','—')}")
    rep.append(f"Sessions: {d['usage']['sessions']} (avg. {d['usage']['avg_session_mins']} mins)")
    rep.append(f"Completion: {d['usage']['lessons_done']} of {d['usage']['lessons_total']} lessons ({d['usage']['completion_pct']}%)")
    rep.append(f"Trend: {d['usage'].get('trend_vs_prev_pct',0)}% vs last period\n")

    rep.append("4) Focus & Concentration")
    rep.append(f"Summary: {'Improved attention relative to class median.' if d['focus']['focus_score'] >= d['focus'].get('class_median', 62) else 'Below class median; consider shorter, more frequent sessions.'}")
    rep.append(f"Focus score: {d['focus']['focus_score']} (class median: {d['focus'].get('class_median','—')})")
    rep.append(f"Avg. attention block: {d['focus'].get('avg_sustained_block_mins','—')} mins\n")

    rep.append("5) Learning Progress & Mastery")
    rep.append("Summary: Subject-level growth based on activity performance.")
    for s in d["learning"].get("skills", []):
        rep.append(f"- {s['name']}: {s['value']:.2f} ({s['delta']:+.02f})")
    rep.append(f"Perseverance index: {d['learning'].get('perseverance_index','—')} (fraction of attempts using hints)\n")

    rep.append("6) Reading, Language & Expression")
    rep.append("Summary: Not available in this dataset.")
    rep.append("Readability: —")
    rep.append("TTR: —\n")

    rep.append("7) AI Interaction Quality & Support Usage")
    rep.append("Summary: Derived hints usage (no built-in AI support fields in source).")
    rep.append(f"Hints used per attempt: {d['ai_support'].get('hints_per_activity','—')}\n")

    rep.append("8) Motivation & Routine")
    rep.append(f"Summary: {'Low drop-off risk.' if d['routine'].get('dropoff_risk','low')=='low' else 'Potential drop-off risk.'}\n")

    rep.append("9) Technology & Accessibility Diagnostics")
    rep.append("Summary: Device info is partial in this dataset.\n")

    rep.append("10) Goals & Recommendations")
    rep.append("Recommendations:")
    for r in d.get("recommendations", []):
        rep.append(f"- {r}")
    if not d.get("recommendations"):
        rep.append("- Encourage regular short practice sessions (5–7 mins) on weaker subjects")
        rep.append("- Review missed questions in recent attempts")
        rep.append("- Use shorter sessions if average session length is below 10 mins")
    rep.append("")

    rep.append("11) Unanswered & Out-of-Scope Questions")
    rep.append("Summary: Not tracked in this dataset.")
    rep.append("Total questions: —")
    rep.append("Unanswered: — | Out-of-scope: —")
    return "\n".join(rep).strip()


# ---------- Full pie helper (Plotly) ----------
def donut_chart(done: int, total: int, title: str):
    """
    Full pie (hole=0). Returns a plotly.graph_objects.Figure.
    """
    total = int(max(1, total))
    done = int(max(0, min(done, total)))
    remaining = total - done
    labels = ["Done", "Remaining"]
    values = [done, remaining]
    colors = ["#2E86AB", "#E5ECF6"]  # professional blue + soft gray

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0,                 # full pie
                sort=False,
                direction="clockwise",
                marker=dict(colors=colors, line=dict(color="white", width=1)),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value} of " + str(total) + "<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        height=240,
        margin=dict(l=10, r=10, t=50, b=10),
        title=f"{title}\n{done} / {total}",
        showlegend=False,
    )
    return fig


# ---------- Mini meter helper (Altair) ----------
def meter_chart(value: float, max_value: float, title: str, unit: str = "", fmt: str = ".1f") -> alt.Chart:
    max_value = float(max(1.0, max_value, value))
    df = pd.DataFrame([{"name": title, "value": float(value)}])
    bar = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title=None, scale=alt.Scale(domain=[0, max_value])),
            y=alt.Y("name:N", title=None, axis=None),
            tooltip=[alt.Tooltip("value:Q", format=fmt)],
        )
        .properties(height=60, title=f"{title}: {value:{fmt}}{unit}")
    )
    text = bar.mark_text(align="left", dx=3, dy=0).encode(text=alt.Text("value:Q", format=fmt))
    return bar + text


# ---------- Personalisation extraction & charts ----------
AVATAR_KEYS = ["avatar", "avatar_name", "selected_avatar", "active_avatar", "avatarId", "avatar_id"]
FONT_KEYS = ["font", "font_name", "selected_font", "text_font"]
BACKGROUND_KEYS = ["background", "background_name", "background_theme", "bg_theme", "bg", "selected_background"]


def _extract_attr(record: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    # exact match first
    for k in candidates:
        if k in record and record[k] not in (None, ""):
            return str(record[k])
    # fallback: any key that contains the token
    for key in record.keys():
        lk = key.lower()
        for c in candidates:
            if c.lower() in lk and record.get(key) not in (None, ""):
                return str(record[key])
    return None


def collect_personalisation_usage(data: Dict[str, Any], user_id: int,
                                  start_dt: Optional[datetime],
                                  end_dt: Optional[datetime]) -> Dict[str, Dict[str, float]]:
    """Sum time_spent (minutes) by avatar/font/background within an optional period."""
    idx = build_indexes(data)
    usage = {"avatar": {}, "font": {}, "background": {}}

    def _within(r: Dict[str, Any], keys: List[str]) -> bool:
        if not start_dt or not end_dt:
            return True
        ts = pick_first_ts(r, keys)
        return (ts is not None) and (start_dt <= ts <= end_dt)

    def _add(kind: str, label: Optional[str], minutes: float):
        if not label or minutes <= 0:
            return
        usage[kind][label] = usage[kind].get(label, 0.0) + minutes

    # topic_session
    for r in data.get("topic_session", []):
        if topic_session_user_id(r, idx) != user_id:
            continue
        if not _within(r, ["started_at", "completed_at", "created_at", "timestamp", "date"]):
            continue
        mins = get_time_spent(r)
        _add("avatar", _extract_attr(r, AVATAR_KEYS), mins)
        _add("font", _extract_attr(r, FONT_KEYS), mins)
        _add("background", _extract_attr(r, BACKGROUND_KEYS), mins)

    # chapter_session
    for r in data.get("chapter_session", []):
        if chapter_session_user_id(r, idx) != user_id:
            continue
        if not _within(r, ["started_at", "completed_at", "created_at", "timestamp", "date"]):
            continue
        mins = get_time_spent(r)
        _add("avatar", _extract_attr(r, AVATAR_KEYS), mins)
        _add("font", _extract_attr(r, FONT_KEYS), mins)
        _add("background", _extract_attr(r, BACKGROUND_KEYS), mins)

    # lesson_session
    for r in data.get("lesson_session", []):
        if r.get("user_id") != user_id:
            continue
        if not _within(r, ["created_at", "started_at", "completed_at", "timestamp", "date"]):
            continue
        mins = get_time_spent(r)
        _add("avatar", _extract_attr(r, AVATAR_KEYS), mins)
        _add("font", _extract_attr(r, FONT_KEYS), mins)
        _add("background", _extract_attr(r, BACKGROUND_KEYS), mins)

    # daily_activity_log
    for r in data.get("daily_activity_log", []):
        if r.get("user_id") != user_id:
            continue
        if not _within(r, ["login_timestamp", "created_at", "timestamp", "date"]):
            continue
        mins = get_time_spent(r)
        _add("avatar", _extract_attr(r, AVATAR_KEYS), mins)
        _add("font", _extract_attr(r, FONT_KEYS), mins)
        _add("background", _extract_attr(r, BACKGROUND_KEYS), mins)

    # activity_performance (optional time_spent)
    for r in data.get("activity_performance", []):
        if perf_user_id(r, idx) != user_id:
            continue
        if not _within(r, ["submitted_at"]):
            continue
        mins = get_time_spent(r)  # many datasets have 0 here; fine to skip if so
        _add("avatar", _extract_attr(r, AVATAR_KEYS), mins)
        _add("font", _extract_attr(r, FONT_KEYS), mins)
        _add("background", _extract_attr(r, BACKGROUND_KEYS), mins)

    return usage


def _usage_df(usage: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for kind, d in usage.items():
        for name, mins in d.items():
            rows.append({"type": kind, "name": name, "minutes": round(float(mins), 1)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["type", "minutes"], ascending=[True, False])
    return df


def _usage_chart(df: pd.DataFrame, kind: str, title_suffix: str) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame([{"msg": "No usage"}])).mark_text().encode(text="msg:N")
    sub = df[df["type"] == kind].copy()
    if sub.empty:
        return alt.Chart(pd.DataFrame([{"msg": f"No {kind} usage"}])).mark_text().encode(text="msg:N")
    # keep top 8 for readability
    sub = sub.nlargest(8, "minutes")
    chart = (
        alt.Chart(sub)
        .mark_bar()
        .encode(
            x=alt.X("minutes:Q", title="Minutes"),
            y=alt.Y("name:N", sort="-x", title=""),
            tooltip=["name:N", alt.Tooltip("minutes:Q", format=".1f")],
        )
        .properties(height=180, title=f"{kind.title()} usage — {title_suffix}")
    )
    return chart


def render_personalisation_usage(data: Dict[str, Any], user_id: int,
                                 start_dt: datetime, end_dt: datetime):
    st.subheader("🎭 Personalisation usage (Avatar • Font • Background)")

    tabs = st.tabs(["This period", "All time"])

    # Period
    with tabs[0]:
        usage_p = collect_personalisation_usage(data, user_id, start_dt, end_dt)
        df_p = _usage_df(usage_p)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.altair_chart(_usage_chart(df_p, "avatar", "period"), use_container_width=True)
        with c2:
            st.altair_chart(_usage_chart(df_p, "font", "period"), use_container_width=True)
        with c3:
            st.altair_chart(_usage_chart(df_p, "background", "period"), use_container_width=True)

        if not df_p.empty:
            st.download_button("Download period personalisation (.csv)",
                               df_p.to_csv(index=False).encode("utf-8"),
                               file_name=f"user_{user_id}_personalisation_period.csv",
                               mime="text/csv")
        else:
            st.info("No personalisation usage found in the selected period.")

    # All-time
    with tabs[1]:
        usage_all = collect_personalisation_usage(data, user_id, None, None)
        df_a = _usage_df(usage_all)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.altair_chart(_usage_chart(df_a, "avatar", "all-time"), use_container_width=True)
        with c2:
            st.altair_chart(_usage_chart(df_a, "font", "all-time"), use_container_width=True)
        with c3:
            st.altair_chart(_usage_chart(df_a, "background", "all-time"), use_container_width=True)

        if not df_a.empty:
            st.download_button("Download all-time personalisation (.csv)",
                               df_a.to_csv(index=False).encode("utf-8"),
                               file_name=f"user_{user_id}_personalisation_alltime.csv",
                               mime="text/csv")
        else:
            st.info("No personalisation usage found in the dataset.")


# ---------- Subject charts & logs ----------
def render_subject_growth(agg: Dict[str, Any]):
    st.subheader("📈 Subject Growth (derived from activity_performance)")
    if not agg.get("subject_series"):
        st.write("No subject-level performance history available for this user in the selected period or dataset.")
        return

    tabs = st.tabs(list(agg["subject_series"].keys()))
    for tab_obj, subject in zip(tabs, agg["subject_series"].keys()):
        with tab_obj:
            hist = agg["subject_series"][subject]
            if not hist:
                st.write("No data.")
                continue
            df = pd.DataFrame(hist, columns=["date", "score"])
            try:
                df["date"] = pd.to_datetime(df["date"])
            except Exception:
                pass
            chart = alt.Chart(df).mark_line(point=True).encode(
                x="date:T", y="score:Q"
            ).properties(height=200, title=f"{subject} — score over time")
            st.altair_chart(chart, use_container_width=True)
            if len(df) >= 2 and float(df["score"].iloc[0]) != 0:
                try:
                    pct_imp = (df["score"].iloc[-1] - df["score"].iloc[0]) / abs(df["score"].iloc[0]) * 100
                    st.write(f"% improvement: {pct_imp:.0f}% from {df['date'].iloc[0].date()}")
                except Exception:
                    pass


def render_event_log_table(data: Dict[str, Any], user_id: int):
    st.subheader("🧾 Per-event log (joined)")

    idx = build_indexes(data)
    rows: List[Dict[str, Any]] = []

    def _val(r, keys):
        return _extract_attr(r, keys) or "—"

    # Daily logs
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

    # Lesson sessions
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

    # Topic sessions
    for ts in data.get("topic_session", []):
        uid = topic_session_user_id(ts, idx)
        if uid != user_id:
            continue
        en = idx["enrollment_by_id"].get(ts.get("enrollment_id"))
        subj = "—"
        if en:
            topic = idx["topics_by_id"].get(en.get("topic_id"))
            if topic:
                subj = topic.get("subject", "—")
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

    # Chapter sessions
    for cs in data.get("chapter_session", []):
        uid = chapter_session_user_id(cs, idx)
        if uid != user_id:
            continue
        ts = idx["topic_session_by_id"].get(cs.get("topic_session_id"))
        subj = "—"
        if ts:
            en = idx["enrollment_by_id"].get(ts.get("enrollment_id"))
            if en:
                topic = idx["topics_by_id"].get(en.get("topic_id"))
                if topic:
                    subj = topic.get("subject", "—")
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

    # Activity performance rows
    for ap in data.get("activity_performance", []):
        uid = perf_user_id(ap, idx)
        if uid != user_id:
            continue
        subj = perf_subject(ap, idx) or "—"
        score_val = ap.get("score", None)
        if score_val in (None, ""):
            score_val = 100.0 if ap.get("is_right") else 0.0
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


# ---------- Streamlit main ----------
def main():
    st.title("Student Report Generator (from raw events)")
    st.markdown("Type a query like: `give summary about user_id 8 for teacher`.")

    data_path = st.sidebar.text_input("JSON data path", value=DEFAULT_JSON_PATH)
    query = st.text_input("Query", value="give summary about user_id 8")

    if not Path(data_path).exists():
        st.error(f"JSON file not found at {data_path}")
        return
    try:
        data = load_json(data_path)
    except Exception as e:
        st.error(f"Failed to load JSON: {e}")
        return

    user_id, audience = extract_user_id_and_audience(query)
    if user_id is None:
        st.info("Could not extract `user_id` from query. Use syntax like 'user_id 8'.")
        users = pd.DataFrame(data.get("user", []))
        if not users.empty:
            cols = [c for c in ["user_id", "name", "email", "class_level"] if c in users.columns]
            st.write("Users in dataset:")
            st.table(users[cols])
        return

    idx = build_indexes(data)
    rec_start, rec_end = available_date_range_for_user(data, user_id, idx)

    today = date.today()
    default_end = rec_end.date() if rec_end else today
    default_start = rec_start.date() if rec_start else (default_end - timedelta(days=6))

    st.sidebar.caption("Pick a date range that overlaps your user's events.")
    start_date = st.sidebar.date_input("Report start date", value=default_start)
    end_date = st.sidebar.date_input("Report end date", value=default_end)

    if rec_start and rec_end:
        st.success(f"Recommended date range for user {user_id}: **{rec_start.date()} → {rec_end.date()}** (covers all their events).")

    if st.button("Run"):
        try:
            agg = aggregate_student(data, user_id)
        except Exception as e:
            st.error(f"Aggregation error: {e}")
            return

        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        curr = period_stats(data, user_id, start_dt, end_dt)

        prev_span_days = max(1, (end_dt.date() - start_dt.date()).days + 1)
        prev_end = start_dt - timedelta(seconds=1)
        prev_start = prev_end - timedelta(days=prev_span_days - 1)
        prev = period_stats(data, user_id, prev_start, prev_end)

        trend_vs_prev = compute_trend(curr["total_time_mins"], prev["total_time_mins"])

        # ---- User metadata ----
        display_user_metadata(agg)

        # ============================================================
        # SNAPSHOT CONTAINER (two stacked rows inside one container)
        # ============================================================
        with st.container():
            st.subheader(f"Engagement & Performance Snapshot for {agg['name']} (ID {user_id})")

            # Row 1: Average score — left % text, right pie
            r1_left, r1_right = st.columns([1, 1])
            with r1_left:
                st.metric("Average activity score (period)", f"{curr['avg_score']}%")
                st.caption(f"Active days: {curr.get('active_days','—')}")
            with r1_right:
                score_val = max(0, min(100, float(curr.get("avg_score", 0))))
                labels = ["Score", "Remaining"]
                values = [score_val, max(0.0, 100.0 - score_val)]
                colors = ["#2E86AB", "#E5ECF6"]
                fig_score = go.Figure(
                    data=[
                        go.Pie(
                            labels=labels,
                            values=values,
                            hole=0,  # full pie
                            sort=False,
                            direction="clockwise",
                            marker=dict(colors=colors, line=dict(color="white", width=1)),
                            textinfo="label+percent",
                            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
                        )
                    ]
                )
                fig_score.update_layout(
                    template="plotly_white",
                    height=240,
                    margin=dict(l=10, r=10, t=50, b=10),
                    title="Average activity score (period)",
                    showlegend=False,
                )
                st.plotly_chart(fig_score, use_container_width=True)

            st.divider()

            # Row 2 (bottom of first container): Completion — left % text, right pie
            r2_left, r2_right = st.columns([1, 1])
            with r2_left:
                st.metric("Lesson completion (period)", f"{curr['completion_pct']}%")
                st.caption(f"{curr['lessons_done']} / {curr['lessons_total']} topics ≥80%")
            with r2_right:
                st.plotly_chart(
                    donut_chart(curr["lessons_done"], curr["lessons_total"], "Lessons completed (topics ≥80%)"),
                    use_container_width=True
                )

                # ----- Period KPI charts -----
            st.markdown("### Period KPI charts")
            pc1, pc2 = st.columns(2)

            with pc1:
                st.plotly_chart(
                    pie_for_period_kpi("Avg session length (period)", curr["avg_session_mins"], cap=50, unit=" mins"),
                    use_container_width=True
                )

            with pc2:
                st.plotly_chart(
                    pie_for_period_kpi("Time-on-task (period)", curr["total_time_mins"], cap=100, unit=" mins"),
                    use_container_width=True
                )


        pc3, pc4 = st.columns(2)
        with pc3:
            st.altair_chart(
                meter_chart(curr["total_time_mins"], max_value=max(120, curr["total_time_mins"] * 1.4),
                            title="Total time (period)", unit=" mins"),
                use_container_width=True
            )
        with pc4:
            st.altair_chart(
                meter_chart(curr["sessions"], max_value=max(10, curr["sessions"] * 1.4),
                            title="Sessions counted (period)"),
                use_container_width=True
            )

        # KPI recap
        c1, c2, c3 = st.columns(3)
        c1.write(f"**Lessons completed:** {curr['lessons_done']} / {curr['lessons_total']} ({curr['completion_pct']}%)")
        c2.write(f"**Total time:** {curr['total_time_mins']} minutes")
        c3.write(f"**Sessions counted:** {curr['sessions']}")
        st.write(f"**All-time points:** {agg['total_points']} | **All-time avg session:** {agg['avg_session_length']} mins")
        st.write(f"**All-time chapter progress:** {agg['chapters_seen']} chapters seen, average progress {agg['avg_chapter_progress_val']}%")
        st.write(f"**All-time hints usage:** {agg['avg_hints_used']} of attempts used a hint (0..1)")

                # ---------- All-time KPI charts ----------
        st.markdown("### All-time KPI charts")

        # Top number tiles (keep)
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("All-time points", f"{agg['total_points']:.1f}")
        t2.metric("All-time avg session", f"{agg['avg_session_length']:.1f} mins")
        t3.metric(f"Avg chapter progress ({agg['chapters_seen']} seen)", f"{agg['avg_chapter_progress_val']:.1f} %")
        t4.metric("Hints usage (all-time)", f"{agg['avg_hints_used']*100:.1f} %")

        # Combined vertical bar chart with 4 bars
        labels = [
            "All-time points",
            "All-time avg session (mins)",
            f"Avg chapter progress ({agg['chapters_seen']} seen) (%)",
            "Hints usage (all-time) (%)",
        ]
        values = [
            float(agg["total_points"]),
            float(agg["avg_session_length"]),
            float(agg["avg_chapter_progress_val"]),
            float(agg["avg_hints_used"] * 100.0),
        ]
        units = ["", " mins", " %", " %"]
        colors = ["#2E86AB", "#6AA84F", "#FF8C42", "#B35C9E"]
        texts = [f"{v:.1f}{u}" if u else f"{v:.1f}" for v, u in zip(values, units)]

        fig_kpi = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=colors,
                    text=texts,
                    textposition="outside",   # show values above bars
                    cliponaxis=False,         # allow text to sit outside plot
                    hovertemplate="%{x}: %{y:.1f}%{customdata}<extra></extra>",
                    customdata=units,
                )
            ]
        )
        fig_kpi.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="",
            yaxis_title="",
            xaxis=dict(tickangle=-10, categoryorder="array", categoryarray=labels),
            yaxis=dict(showgrid=True, zeroline=True, rangemode="tozero"),
            uniformtext_minsize=10,
            uniformtext_mode="show",
            showlegend=False,
        )
        st.plotly_chart(fig_kpi, use_container_width=True)

        # CSV download (unchanged)
        kpi_df = pd.DataFrame([
            {"metric": "All-time points", "value": agg["total_points"], "unit": ""},
            {"metric": "All-time avg session", "value": agg["avg_session_length"], "unit": "mins"},
            {"metric": f"Avg chapter progress ({agg['chapters_seen']} seen)", "value": agg["avg_chapter_progress_val"], "unit": "%"},
            {"metric": "Hints usage (all-time)", "value": agg["avg_hints_used"] * 100.0, "unit": "%"},
        ])
        st.download_button("Download all-time KPIs (.csv)",
                           kpi_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"user_{user_id}_alltime_kpis.csv",
                           mime="text/csv")

        # ---- Subject charts ----
        render_subject_growth(agg)

        # ---------- Personalisation usage (above SEN Report) ----------
        render_personalisation_usage(data, user_id, start_dt, end_dt)

        # ---------- SEN Report ----------
        st.subheader("🧾 SEN Report (auto-generated)")
        if not curr["had_ts"]:
            st.warning("No reliable timestamps found **in your selected range**. Use the recommended range above for complete figures.")

        focus_score_now = compute_focus_score(curr["completion_pct"], curr["avg_session_mins"])
        focus_score_prev = compute_focus_score(prev["completion_pct"], prev["avg_session_mins"])
        focus_delta = focus_score_now - focus_score_prev

        subject_to_scores_curr: Dict[str, List[float]] = {}
        subject_to_scores_prev: Dict[str, List[float]] = {}
        idx_local = build_indexes(data)

        def perf_rows_in_range(start_dt, end_dt):
            rows = []
            for ap in data.get("activity_performance", []):
                if perf_user_id(ap, idx_local) != user_id:
                    continue
                dt = parse_ts(ap.get("submitted_at"))
                if dt and (start_dt <= dt <= end_dt):
                    rows.append(ap)
            return rows

        prs_curr = perf_rows_in_range(start_dt, end_dt)
        prs_prev = perf_rows_in_range(prev_start, prev_end)

        for row, bucket in [(prs_curr, subject_to_scores_curr), (prs_prev, subject_to_scores_prev)]:
            for ap in row:
                subj = perf_subject(ap, idx_local) or "Unknown"
                score_val = float(ap.get("score")) if ap.get("score") not in (None, "") else (100.0 if ap.get("is_right") else 0.0)
                bucket.setdefault(subj, []).append(score_val)

        skills = []
        for subj, vals in subject_to_scores_curr.items():
            v_now = mean(vals) if vals else 0.0
            v_prev = mean(subject_to_scores_prev.get(subj, [])) if subject_to_scores_prev.get(subj) else 0.0
            skills.append({"name": subj, "value": v_now/100.0, "delta": (v_now - v_prev)/100.0})

        dropoff_risk = "high" if (curr["active_days"] <= 2 or curr["completion_pct"] < 30) else ("medium" if curr["completion_pct"] < 60 else "low")

        report_data = {
            "student": {
                "name": agg["name"],
                "id": user_id,
                "class": agg.get("class_level", "—"),
                "year": agg.get("class_level", "—")
            },
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "generated_on": date.today().isoformat()
            },
            "prepared_for": "Teacher" if audience == "teacher" else "Parent/Carer",
            "devices": {},
            "usage": {
                "active_days": curr.get("active_days", "—"),
                "sessions": curr["sessions"],
                "avg_session_mins": curr["avg_session_mins"],
                "lessons_done": curr["lessons_done"],
                "lessons_total": curr["lessons_total"],
                "completion_pct": curr["completion_pct"],
                "total_time_mins": curr["total_time_mins"],
                "trend_vs_prev_pct": trend_vs_prev
            },
            "focus": {
                "focus_score": focus_score_now,
                "focus_score_delta": focus_delta,
                "class_median": 62,
                "avg_sustained_block_mins": curr["avg_session_mins"],
            },
            "accommodations": {},
            "learning": {
                "skills": skills,
                "perseverance_index": agg.get("avg_hints_used", "—"),
            },
            "language": {},
            "ai_support": {
                "hints_per_activity": agg.get("avg_hints_used", "—"),
            },
            "routine": {
                "dropoff_risk": dropoff_risk
            },
            "goals": [],
            "recommendations": [],
            "questions": {}
        }

        report_text = build_report(report_data)
        st.text_area("Report (copy-ready)", value=report_text, height=600)
        st.download_button(
            "Download report (.txt)",
            data=report_text.encode("utf-8"),
            file_name=f"sen_report_user_{user_id}_{start_date}_{end_date}.txt",
            mime="text/plain"
        )

        # Joined event log
        render_event_log_table(data, user_id)


def display_user_metadata(agg: Dict[str, Any]):
    with st.expander("User Profile & Metadata", expanded=True):
        cols = st.columns([1, 3])
        if agg.get("avatar"):
            try:
                cols[0].image(agg["avatar"], width=100)
            except Exception:
                pass
        info = (
            f"**Name:** {agg['name']}\n\n"
            f"**Age:** {agg['age_display']}\n\n"
            f"**Gender:** {agg['gender']}\n\n"
            f"**Email:** {agg['email']}\n\n"
            f"**Parental Email:** {agg['parental_email']}\n\n"
            f"**School:** {agg['school_name']}, **Class Level:** {agg['class_level']}, "
            f"**Reading Level:** {agg['reading_level']}\n\n"
            f"**Account Created:** {agg['created_at']}, **Last Updated:** {agg['updated_at']}\n\n"
            f"**All-time points:** {agg['total_points']} | **Avg session:** {agg['avg_session_length']} mins"
        )
        cols[1].markdown(info)


if __name__ == "__main__":
    main()
