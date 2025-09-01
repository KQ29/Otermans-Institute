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

# ---------- Defaults ----------
DEFAULT_JSON_PATH = "student_fake_data.json"

st.set_page_config(page_title="Student Report Generator", layout="wide")


# ---------- Utilities ----------
def parse_ts(ts: Any) -> Optional[datetime]:
    """Parse many common timestamp formats. Returns None if not parseable."""
    if ts is None:
        return None
    if isinstance(ts, (datetime, pd.Timestamp)):
        return pd.to_datetime(ts).to_pydatetime()
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
        return pd.to_datetime(s, errors="coerce").to_pydatetime()
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
    """Use either total_time_spent or time_spent if present."""
    v = r.get("total_time_spent", None)
    if v is None:
        v = r.get("time_spent", 0)
    return float(v or 0.0)


def compute_age_from_dob(dob_str: str) -> str:
    """Return years as a string, or '—' if not available/invalid."""
    if not dob_str:
        return "—"
    # Try multiple common formats
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            dob = datetime.strptime(dob_str, fmt).date()
            break
        except Exception:
            dob = None
    if dob is None:
        try:
            dob = datetime.fromisoformat(dob_str).date()
        except Exception:
            return "—"
    today = date.today()
    years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return str(years)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_user_id_and_audience(query: str):
    m = re.search(r'user[_\s]?id\s*[:=]?\s*(\d+)', query, flags=re.IGNORECASE)
    user_id = int(m.group(1)) if m else None
    audience = "parent"
    if re.search(r'\bteacher\b', query, flags=re.IGNORECASE):
        audience = "teacher"
    return user_id, audience


# ---------- Joining helpers (built for student_fake_data.json) ----------
def build_indexes(data: Dict[str, Any]):
    enrollment_by_id = {e["enrollment_id"]: e for e in data.get("enrollment", [])}
    topics_by_id = {t["topic_id"]: t for t in data.get("topics", [])}
    topic_session_by_id = {t["topic_session_id"]: t for t in data.get("topic_session", [])}
    chapter_sessions = data.get("chapter_session", [])
    lesson_sessions = data.get("lesson_session", [])
    activity_perf = data.get("activity_performance", [])
    daily_logs = data.get("daily_activity_log", [])
    users = {u["user_id"]: u for u in data.get("user", [])}
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
    """Map chapter_session -> topic_session -> enrollment -> user_id."""
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
    """Map activity_performance -> chapter_session -> user_id."""
    cs_id = ap.get("chapter_session_id")
    cs = next((c for c in idx["chapter_sessions"] if c.get("chapter_session_id") == cs_id), None)
    if not cs:
        return None
    return chapter_session_user_id(cs, idx)


def perf_subject(ap: Dict[str, Any], idx: Dict[str, Any]) -> Optional[str]:
    """Get subject from performance row through enrollment -> topic."""
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

    # Points and time from multiple sources (sum across)
    total_points = 0.0
    total_time = 0.0
    all_session_lengths: List[float] = []

    # Topic sessions (map through enrollment -> user)
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

    # Chapter sessions (map through topic_session -> enrollment -> user)
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

    # Lesson sessions (standalone; keep for metadata/time)
    lesson_sessions = [l for l in idx["lesson_sessions"] if l.get("user_id") == user_id]
    for l in lesson_sessions:
        t = get_time_spent(l)
        total_time += t
        if t > 0:
            all_session_lengths.append(t)

    # Performance rows (scores + hint usage)
    perf_rows = [ap for ap in idx["activity_perf"] if perf_user_id(ap, idx) == user_id]
    scores = [float(p.get("score") or 0.0) for p in perf_rows]
    avg_score = mean(scores) if scores else 0.0

    hints_used = [1.0 if (p.get("used_hint") in (True, 1, "true", "True")) else 0.0 for p in perf_rows]
    avg_hints_used = mean(hints_used) if hints_used else 0.0  # 0..1 fraction of attempts with a hint

    # Completion: treat "topic completed" as topic_session.completion_percent >= 80 (tunable)
    topics_completed = sum(1 for ts in ts_for_user if float(ts.get("completion_percent") or 0) >= 80)
    topics_total = len(ts_for_user)
    lesson_completion_rate = pct(topics_completed, topics_total)

    # Chapter progress (average)
    ch_progress = [float(c.get("progress_percent") or 0) for c in cs_for_user]
    avg_chapter_progress = mean(ch_progress) if ch_progress else 0.0
    chapter_progress_summary = f"{len(cs_for_user)} chapters seen, average progress {avg_chapter_progress:.1f}%"

    avg_session_length = mean(all_session_lengths) if all_session_lengths else 0.0

    # Subject growth series (from performance rows)
    per_subject_series: Dict[str, List[Tuple[str, float]]] = {}
    for ap in perf_rows:
        subj = perf_subject(ap, idx) or "Unknown"
        dt = parse_ts(ap.get("submitted_at"))
        if dt:
            per_subject_series.setdefault(subj, []).append((dt.date().isoformat(), float(ap.get("score") or 0)))

    # User profile basics
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
        "avg_hints_used": round(avg_hints_used, 3),  # 0..1 fraction
        "dob": dob or "Unknown",
        "age_display": age_display,
        "gender": gender,
        "email": email,
        "parental_email": parental_email,
        "onboarding_complete": user.get("is_onboarding_complete", False),
        "created_at": user.get("created_at", "Unknown"),
        "updated_at": user.get("updated_at", "Unknown"),
        "avatar": avatar,

        # Per-subject time series derived from events
        "subject_series": per_subject_series,

        # For tables/charts
        "ts_for_user": ts_for_user,
        "cs_for_user": cs_for_user,
        "lesson_sessions": lesson_sessions,
        "perf_rows": perf_rows,
    }
    return aggregated


# ---------- Period-based metrics (using REAL keys in your JSON) ----------
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
    # JSON uses: started_at, completed_at, submitted_at, login_timestamp, created_at
    ts_keys_sessions = ["started_at", "completed_at", "created_at", "timestamp", "date"]
    ts_keys_perf = ["submitted_at"]
    ts_keys_dailies = ["login_timestamp", "created_at", "timestamp", "date"]

    idx = build_indexes(data)

    # prepare collections already mapped to the selected user
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

    # Totals and averages
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

    # Define "lessons" as topic_sessions, "done" when completion_percent >= 80
    completed = sum(1 for t in topic_sessions if float(t.get("completion_percent") or 0) >= 80)
    total_lessons = len(topic_sessions)
    completion_pct = pct(completed, total_lessons)

    # Active days (unique dates seen across any event)
    day_set = set()
    # unify helper for dates
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

    # Average score in the period
    scores = [float(p.get("score") or 0) for p in perf_rows]
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
        "avg_score": round(avg_score, 1)
    }


def compute_trend(curr: float, prev: float) -> int:
    if prev <= 0:
        return 0
    return int(round(100 * (curr - prev) / prev))


def compute_focus_score(completion_pct: float, avg_session_mins: float) -> int:
    """Lightweight focus score (0..100) using completion and avg session length."""
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
        rows.append({
            "event": "activity_attempt",
            "timestamp": ap.get("submitted_at"),
            "subject": subj,
            "score": ap.get("score", "—"),
            "points": ap.get("points_earned", 0),
            "time_spent": "—",
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

    # Inputs
    data_path = st.sidebar.text_input("JSON data path", value=DEFAULT_JSON_PATH)

    # We let the user type query first, so we can set a smart default range later
    query = st.text_input("Query", value="give summary about user_id 8")

    # Load data (once)
    if not Path(data_path).exists():
        st.error(f"JSON file not found at {data_path}")
        return
    try:
        data = load_json(data_path)
    except Exception as e:
        st.error(f"Failed to load JSON: {e}")
        return

    # Parse user/audience
    user_id, audience = extract_user_id_and_audience(query)
    if user_id is None:
        st.info("Could not extract `user_id` from query. Use syntax like 'user_id 8'.")
        # Show users present so the user can pick
        users = pd.DataFrame(data.get("user", []))
        if not users.empty:
            st.write("Users in dataset:")
            st.table(users[["user_id", "name", "email", "class_level"]])
        return

    idx = build_indexes(data)

    # Recommended available date range (full extent of events for this user)
    rec_start, rec_end = available_date_range_for_user(data, user_id, idx)

    # Date inputs (default to recommended range if available, else last 7 days)
    today = date.today()
    if rec_end:
        default_end = rec_end.date()
    else:
        default_end = today
    if rec_start:
        default_start = rec_start.date()
    else:
        default_start = default_end - timedelta(days=6)

    st.sidebar.caption("Pick a date range that overlaps your user's events.")
    start_date = st.sidebar.date_input("Report start date", value=default_start)
    end_date = st.sidebar.date_input("Report end date", value=default_end)

    if rec_start and rec_end:
        st.success(f"Recommended date range for user {user_id}: **{rec_start.date()} → {rec_end.date()}** "
                   f"(covers all their events).")

    if st.button("Run"):
        try:
            agg = aggregate_student(data, user_id)
        except Exception as e:
            st.error(f"Aggregation error: {e}")
            return

        # Per-period snapshot (+ previous period for trend)
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

        # ---- Engagement & Performance Snapshot ----
        st.subheader(f"Engagement & Performance Snapshot for {agg['name']} (ID {user_id})")

        mcol, pcol = st.columns([1, 1])
        with mcol:
            st.metric("Average activity score", f"{curr['avg_score']}%")
            st.metric("Avg session length", f"{curr['avg_session_mins']} mins")

        with pcol:
            score_val = max(0, min(100, float(curr.get("avg_score", 0))))
            pie_df = pd.DataFrame(
                [{"label": "Score", "value": score_val},
                 {"label": "Remaining", "value": max(0.0, 100.0 - score_val)}]
            )
            donut = alt.Chart(pie_df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta("value:Q"),
                color=alt.Color("label:N", legend=None),
                tooltip=["label:N", alt.Tooltip("value:Q", format=".1f")]
            ).properties(height=160, width=160, title="Average activity score (period)")
            st.altair_chart(donut, use_container_width=False)

        c1, c2, c3 = st.columns(3)
        c1.write(f"**Lessons completed (topics ≥80%):** {curr['lessons_done']} / {curr['lessons_total']} "
                 f"({curr['completion_pct']}%)")
        c2.write(f"**Total time:** {curr['total_time_mins']} minutes")
        c3.write(f"**Sessions counted:** {curr['sessions']}")
        st.write(f"**All-time points:** {agg['total_points']} | **All-time avg session:** {agg['avg_session_length']} mins")
        st.write(f"**All-time chapter progress:** {agg['chapter_progress_summary']}")
        st.write(f"**Hints usage (all-time):** {agg['avg_hints_used']} of attempts used a hint (0..1)")

        # ---- Subject charts (from performance history) ----
        render_subject_growth(agg)

        # ---------- SEN Report (period-based) ----------
        st.subheader("🧾 SEN Report (auto-generated)")
        if not curr["had_ts"]:
            st.warning("No reliable timestamps found **in your selected range**. "
                       "Use the recommended range above for complete figures.")

        # Focus score & deltas
        focus_score_now = compute_focus_score(curr["completion_pct"], curr["avg_session_mins"])
        focus_score_prev = compute_focus_score(prev["completion_pct"], prev["avg_session_mins"])
        focus_delta = focus_score_now - focus_score_prev

        # Simple “skills” summary derived from subjects: current period average by subject (delta vs previous)
        subject_to_scores_curr: Dict[str, List[float]] = {}
        subject_to_scores_prev: Dict[str, List[float]] = {}
        # Build per-subject from performance rows on-the-fly
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
                bucket.setdefault(subj, []).append(float(ap.get("score") or 0))

        skills = []
        for subj, vals in subject_to_scores_curr.items():
            v_now = mean(vals) if vals else 0.0
            v_prev = mean(subject_to_scores_prev.get(subj, [])) if subject_to_scores_prev.get(subj) else 0.0
            skills.append({"name": subj, "value": v_now/100.0, "delta": (v_now - v_prev)/100.0})

        # Routine risk (simple rule-of-thumb)
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

        # Joined event log (so you can see exactly what's driving the metrics)
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
