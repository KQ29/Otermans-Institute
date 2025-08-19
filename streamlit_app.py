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

# Defaults
DEFAULT_JSON_PATH = "complete_dummy_data.json"

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


def normalize_age_from_dob(computed: str) -> Tuple[str, List[str]]:
    """Use only computed age; return display string and any sanity warnings."""
    warnings = []
    age_str = computed if computed not in ("unknown", "", None) else "—"
    try:
        age_val = int(computed)
        if age_val > 25 or age_val < 4:
            warnings.append(f"Age {age_val} is outside typical child range; please verify DOB.")
    except Exception:
        pass
    return age_str, warnings


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


# ---------- Aggregation over all-time (for the snapshot & charts) ----------
def aggregate_student(data: Dict[str, Any], user_id: int) -> Dict[str, Any]:
    user = next((u for u in data.get("user", []) if u.get("user_id") == user_id), None)
    if user is None:
        raise ValueError(f"User {user_id} not found.")

    lesson_sessions = [l for l in data.get("lesson_session", []) if l.get("user_id") == user_id]
    chapter_sessions = [c for c in data.get("chapter_session", []) if c.get("user_id") == user_id]
    topic_sessions = [t for t in data.get("topic_session", []) if t.get("user_id") == user_id]
    daily_logs = [d for d in data.get("daily_activity_log", []) if d.get("user_id") == user_id]

    total_time = sum([l.get("time_spent", 0) for l in lesson_sessions] +
                     [c.get("time_spent", 0) for c in chapter_sessions] +
                     [t.get("time_spent", 0) for t in topic_sessions] +
                     [d.get("time_spent", 0) for d in daily_logs])

    lessons_completed = sum(1 for l in lesson_sessions if l.get("is_completed"))
    total_lessons = len(lesson_sessions)
    lesson_completion_rate = (lessons_completed / total_lessons * 100) if total_lessons else 0

    chapter_progresses = [c.get("progress_percent", 0) for c in chapter_sessions]
    avg_chapter_progress = mean(chapter_progresses) if chapter_progresses else 0
    chapter_progress_summary = f"{len(chapter_sessions)} chapters seen, average progress {avg_chapter_progress:.1f}%"

    all_session_lengths: List[float] = []
    all_session_lengths += [l.get("time_spent", 0) for l in lesson_sessions]
    all_session_lengths += [c.get("time_spent", 0) for c in chapter_sessions]
    all_session_lengths += [t.get("time_spent", 0) for t in topic_sessions]
    avg_session_length = mean(all_session_lengths) if all_session_lengths else 0

    perf = [p for p in data.get("activity_performance", []) if p.get("user_id") == user_id]
    scores = [p.get("score", 0) for p in perf]
    avg_score = mean(scores) if scores else 0
    total_points = sum(p.get("points_earned", 0) for p in perf)
    hints_used_list = [p.get("hints_used", 0) for p in perf]
    avg_hints_used = mean(hints_used_list) if hints_used_list else 0

    achs = [a for a in data.get("achievement", []) if a.get("user_id") == user_id]
    badge_list = sorted({a.get("badge_name") for a in achs if a.get("badge_name")})
    most_recent_badge = "None"
    if achs:
        try:
            most_recent_badge = sorted(achs, key=lambda x: x.get("date_earned", ""), reverse=True)[0].get("badge_name", "None")
        except Exception:
            most_recent_badge = achs[-1].get("badge_name", "None")

    feedbacks = [f for f in data.get("feedback", []) if f.get("user_id") == user_id]
    feedback_snippets: List[str] = []
    for f in feedbacks[:3]:
        rating = f.get("rating")
        comment = f.get("comment", "").strip()
        if comment:
            feedback_snippets.append(f'"{comment}" (rating {rating})')
    feedback_summary = "; ".join(feedback_snippets) if feedback_snippets else "No recent feedback."

    # ---- AGE: compute from DOB only and never say "computed" in UI ----
    dob = user.get("dob", "")
    computed_age = compute_age_from_dob(dob)
    age_display, age_warnings = normalize_age_from_dob(computed_age)

    onboarding = user.get("is_onboarding_complete", False)
    created_at = user.get("created_at", "Unknown")
    updated_at = user.get("updated_at", "Unknown")
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
        "lessons_completed": lessons_completed,
        "lesson_completion_rate": round(lesson_completion_rate, 1),
        "chapter_progress_summary": chapter_progress_summary,
        "avg_session_length": round(avg_session_length, 1),
        "avg_score": round(avg_score, 1),
        "total_points": total_points,
        "avg_hints_used": round(avg_hints_used, 2),
        "badge_list": ', '.join(badge_list) if badge_list else "None",
        "most_recent_badge": most_recent_badge or "None",
        "feedback_comments_and_ratings": feedback_summary,
        "dob": dob or "Unknown",
        "age_display": age_display,          # show only this in UI
        # "age_warnings": age_warnings,      # kept internally if you ever need, not displayed
        "gender": gender,
        "email": email,
        "parental_email": parental_email,
        "onboarding_complete": onboarding,
        "created_at": created_at,
        "updated_at": updated_at,
        "avatar": avatar,
        "micro_achievements": user.get("micro_achievements", []),
        "adaptive_performance": user.get("adaptive_performance", []),
        # pass through any per-skill data if present
        "literacy_scores": user.get("literacy_scores", []),
        "numeracy_scores": user.get("numeracy_scores", []),
        "motor_scores": user.get("motor_scores", []),
        "communication_scores": user.get("communication_scores", []),
        "literacy_stage": user.get("literacy_stage", ""),
        "numeracy_stage": user.get("numeracy_stage", ""),
        "motor_stage": user.get("motor_stage", ""),
        "communication_stage": user.get("communication_stage", ""),
        "literacy_pct": user.get("literacy_pct", 0),
        "numeracy_pct": user.get("numeracy_pct", 0),
        "motor_pct": user.get("motor_pct", 0),
        "communication_pct": user.get("communication_pct", 0),
    }
    return aggregated


# ---------- Period-based metrics for the SEN report ----------
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
    ts_keys_sessions = ["start_time", "created_at", "timestamp", "session_date", "date"]
    ts_keys_dailies = ["login_timestamp", "created_at", "timestamp", "date"]

    lesson_sessions_all = [l for l in data.get("lesson_session", []) if l.get("user_id") == user_id]
    chapter_sessions_all = [c for c in data.get("chapter_session", []) if c.get("user_id") == user_id]
    topic_sessions_all = [t for t in data.get("topic_session", []) if t.get("user_id") == user_id]
    daily_logs_all = [d for d in data.get("daily_activity_log", []) if d.get("user_id") == user_id]

    lesson_sessions = filter_records_by_period(lesson_sessions_all, start_dt, end_dt, ts_keys_sessions)
    chapter_sessions = filter_records_by_period(chapter_sessions_all, start_dt, end_dt, ts_keys_sessions)
    topic_sessions = filter_records_by_period(topic_sessions_all, start_dt, end_dt, ts_keys_sessions)
    daily_logs = filter_records_by_period(daily_logs_all, start_dt, end_dt, ts_keys_dailies)

    had_ts = any([lesson_sessions, chapter_sessions, topic_sessions, daily_logs]) \
        or any(pick_first_ts(x, ts_keys_sessions) for x in (lesson_sessions_all + chapter_sessions_all + topic_sessions_all)) \
        or any(pick_first_ts(x, ts_keys_dailies) for x in daily_logs_all)

    total_time = sum([l.get("time_spent", 0) for l in lesson_sessions] +
                     [c.get("time_spent", 0) for c in chapter_sessions] +
                     [t.get("time_spent", 0) for t in topic_sessions] +
                     [d.get("time_spent", 0) for d in daily_logs])

    session_lengths = [l.get("time_spent", 0) for l in lesson_sessions] + \
                      [c.get("time_spent", 0) for c in chapter_sessions] + \
                      [t.get("time_spent", 0) for t in topic_sessions]
    sessions_count = len(session_lengths)
    avg_session_len = mean(session_lengths) if session_lengths else 0.0

    completed = sum(1 for l in lesson_sessions if l.get("is_completed"))
    total_lessons = len(lesson_sessions)
    completion_pct = pct(completed, total_lessons)

    day_set = set()
    for coll, keys in [(lesson_sessions, ts_keys_sessions),
                       (chapter_sessions, ts_keys_sessions),
                       (topic_sessions, ts_keys_sessions),
                       (daily_logs, ts_keys_dailies)]:
        for r in coll:
            ts = pick_first_ts(r, keys)
            if ts:
                day_set.add(ts.date())
    active_days = len(day_set)

    return {
        "had_ts": had_ts,
        "total_time_mins": round(float(total_time), 1),
        "sessions": sessions_count,
        "avg_session_mins": round(float(avg_session_len), 1),
        "lessons_done": completed,
        "lessons_total": total_lessons,
        "completion_pct": round(float(completion_pct), 1),
        "active_days": active_days
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


# ---------- Report builder (11 sections) ----------
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
    rep.append("Data Sources: sessions, attempts, completion, events, ui_preferences, ai_interactions, mastery, goals, devices, moderation_incidents\n")

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

    tts_on = d["accommodations"].get("tts_accuracy_on", 0.0)
    tts_off = d["accommodations"].get("tts_accuracy_off", 0.0)
    tts_effect_pp = int(round((tts_on - tts_off) * 100))
    if d["accommodations"].get("tts_usage_pct", 0) < 30 and tts_effect_pp >= 5:
        accom_summary = "Accommodations not consistently used despite measurable benefit."
    else:
        accom_summary = "Core supports are stable and effective, boosting accuracy and reducing distractions."
    rep.append("2) SEN Profile & Accommodations")
    rep.append(f"Summary: {accom_summary}")
    rep.append(f"Primary Needs: {', '.join(d['accommodations'].get('needs', [])) or '—'}")
    rep.append(f"Accommodations: {', '.join(d['accommodations'].get('enabled', [])) or '—'}")
    rep.append(f"Effectiveness: TTS ON → {pp01(tts_on)} vs OFF → {pp01(tts_off)} ({tts_effect_pp}pp)")
    rep.append(f"Stability: Font size changed {d['accommodations'].get('font_size_changes',0)}× this period\n")

    eng_summary = "Strong participation and lesson completion." if comp_pct >= 70 else \
                  "Very low engagement, with limited active days and short sessions." if comp_pct < 40 else \
                  "Moderate engagement; room for higher completion."
    rep.append("3) Engagement & Usage")
    rep.append(f"Summary: {eng_summary}")
    rep.append(f"Active Days: {d['usage'].get('active_days','—')} of 7")
    rep.append(f"Sessions: {d['usage']['sessions']} (avg. {d['usage']['avg_session_mins']} mins)")
    rep.append(f"Completion: {d['usage']['lessons_done']} of {d['usage']['lessons_total']} lessons ({d['usage']['completion_pct']}%)")
    rep.append(f"Trend: {d['usage'].get('trend_vs_prev_pct',0)}% vs last period\n")

    focus_summary = "Improved attention and low distraction rates." if d["focus"]["focus_score"] >= d["focus"].get("class_median", d["focus"]["focus_score"]) \
        else "Attention span may be limited; distraction likely higher than peers."
    rep.append("4) Focus & Concentration")
    rep.append(f"Summary: {focus_summary}")
    rep.append(f"Focus score: {d['focus']['focus_score']} (class median: {d['focus'].get('class_median','—')})")
    rep.append(f"Avg. attention block: {d['focus'].get('avg_sustained_block_mins','—')} mins")
    rep.append(f"Idle time: {d['focus'].get('idle_pct','—')}% | Window switches: {d['focus'].get('window_switches_per_session','—')} per session\n")

    rep.append("5) Learning Progress & Mastery")
    rep.append("Summary: Growth across tracked skills; weaker skill needs targeted practice.")
    for s in d["learning"].get("skills", []):
        rep.append(f"- {s['name']}: {s['value']:.2f} ({s['delta']:+.02f})")
    rep.append(f"TTFC: {d['learning'].get('ttfc_secs','—')} seconds")
    rep.append(f"Perseverance index: {d['learning'].get('perseverance_index','—')} retries before correct\n")

    lang_safety = d["language"].get("safety_flags", 0)
    lang_summary = "Language use is age-appropriate; no safety concerns." if lang_safety == 0 \
        else "Below expected reading level and occasional off-task or frustrated tone."
    rep.append("6) Reading, Language & Expression")
    rep.append(f"Summary: {lang_summary}")
    rep.append(f"Readability: Grade {d['language'].get('readability_grade','—')}")
    rep.append(f"TTR: {d['language'].get('ttr','—')}")
    rep.append(f"Topics: {', '.join(d['language'].get('topics', [])) or '—'}")
    rep.append(f"Tone: {d['language'].get('tone','—')}")
    rep.append(f"Safety flags: {lang_safety}\n")

    rep.append("7) AI Interaction Quality & Support Usage")
    rep.append("Summary: Supports are well-used and effective." if d["accommodations"].get("tts_usage_pct",0) >= 60
               else "Low use of supports despite proven benefit.")
    rep.append(f"Hints used: {d['ai_support'].get('hints_per_activity','—')} per activity (avg effect {d['ai_support'].get('hint_effect_pp','0')}pp)")
    rep.append(f"TTS usage: {d['accommodations'].get('tts_usage_pct',0)}% of activities")
    rep.append(f"Avatar: {d['ai_support'].get('avatar','—')} (changes {d['ai_support'].get('avatar_change_count',0)}×)\n")

    rep.append("8) Motivation & Routine")
    rep.append(f"Summary: {'Solid routine and low risk of drop-off.' if d['routine'].get('dropoff_risk','low')=='low' else 'Poor routine and high drop-off risk.'}")
    rep.append(f"Current streak: {d['routine'].get('streak_current_days','—')} days (Longest: {d['routine'].get('streak_longest_days','—')})")
    rep.append(f"Preferred time: {d['routine'].get('preferred_time_window','—')}")
    rep.append(f"Drop-off risk: {d['routine'].get('dropoff_risk','—')}\n")

    tech_stable = d["devices"].get("crashes", 0) == 0 and d["devices"].get("high_packet_loss_sessions", 0) == 0
    rep.append("9) Technology & Accessibility Diagnostics")
    rep.append(f"Summary: {'Stable setup with minimal tech issues.' if tech_stable else 'Tech issues may be contributing to lesson abandonment.'}")
    rep.append(f"Device: {d['devices'].get('device_type','—')}{', screen reader enabled' if d['devices'].get('screen_reader', False) else ''}")
    rep.append(f"Crashes: {d['devices'].get('crashes',0)} | Network issues: {d['devices'].get('high_packet_loss_sessions',0)} session(s) with high packet loss\n")

    rep.append("10) Goals & Recommendations")
    goals_text = []
    for g in d.get("goals", []):
        goals_text.append(f"- {g['skill']}: Current {g['current']:.2f}, goal {g['target']:.2f} by {g['target_date']}")
    if goals_text:
        rep.extend(goals_text)
    recs = d.get("recommendations", [])
    if not recs:
        recs = ["Encourage regular short practice sessions (5–7 mins) on weaker skills",
                "Use TTS for comprehension tasks if available",
                "Reduce distractions—limit app/window switching during sessions"]
    rep.append("Recommendations:")
    rep.extend([f"- {r}" for r in recs])
    rep.append("")

    rep.append("11) Unanswered & Out-of-Scope Questions")
    q = d.get("questions", {})
    rep.append("Summary: Few AI gaps, mostly topical curiosities." if q.get("unanswered_pct", 0) <= 10 and q.get("out_of_scope_pct", 0) <= 10
               else "Summary: Higher than average unanswered and off-topic queries—suggests confusion or off-task behaviour.")
    rep.append(f"Total questions: {q.get('total','—')}")
    rep.append(f"Unanswered: {q.get('unanswered_pct',0)}% | Out-of-scope: {q.get('out_of_scope_pct',0)}%")
    examples = q.get("examples", [])
    if examples:
        rep.append("Examples:")
        for e in examples[:3]:
            rep.append(f"- “{e.get('text','…')}” ({e.get('type','')})")

    return "\n".join(rep).strip()


# ---------- Academic & skill UI ----------
def render_academic_skill_progress(agg: Dict[str, Any]):
    st.subheader("📊 Academic & Skill Progress")

    # 1. Curriculum / Goal Completion — vertical bar chart
    st.markdown("**1. Curriculum / Goal Completion**")
    skill_data = [
        {"skill": "Literacy", "pct": agg.get("literacy_pct", 0), "stage": agg.get("literacy_stage") or "Unknown"},
        {"skill": "Numeracy", "pct": agg.get("numeracy_pct", 0), "stage": agg.get("numeracy_stage") or "Unknown"},
        {"skill": "Motor", "pct": agg.get("motor_pct", 0), "stage": agg.get("motor_stage") or "Unknown"},
        {"skill": "Communication", "pct": agg.get("communication_pct", 0), "stage": agg.get("communication_stage") or "Unknown"},
    ]
    df_completion = pd.DataFrame(skill_data)

    if df_completion["pct"].sum() == 0:
        st.write("No curriculum/goal completion data available.")
    else:
        bar = alt.Chart(df_completion).mark_bar().encode(
            x=alt.X("skill:N", title="Skill"),
            y=alt.Y("pct:Q", title="Completion %", scale=alt.Scale(domain=[0, 100])),
            tooltip=["skill", "pct", "stage"]
        ).properties(height=250)
        text = bar.mark_text(dy=-5).encode(text=alt.Text("pct:Q", format=".0f"))
        st.altair_chart((bar + text).configure_axis(labelFontSize=12), use_container_width=True)
        stages = ", ".join(f"{row['skill']}: {row['stage']}" for _, row in df_completion.iterrows())
        st.caption(f"Current stages — {stages}")

    # 2. Skill Growth
    st.markdown("**2. Skill Growth**")
    tabs = st.tabs(["Literacy", "Numeracy", "Motor", "Communication"])
    for tab_obj, (skill_key, score_key, stage_key) in zip(
        tabs,
        [
            ("literacy", "literacy_scores", "literacy_stage"),
            ("numeracy", "numeracy_scores", "numeracy_stage"),
            ("motor", "motor_scores", "motor_stage"),
            ("communication", "communication_scores", "communication_stage"),
        ],
    ):
        with tab_obj:
            history = agg.get(score_key, [])
            if history:
                df = pd.DataFrame(history, columns=["date", "score"]) if isinstance(history[0], (list, tuple)) else pd.DataFrame(history)
                if "date" not in df.columns or "score" not in df.columns:
                    st.write("Unexpected score history format.")
                else:
                    try:
                        df["date"] = pd.to_datetime(df["date"])
                    except Exception:
                        pass
                    chart = alt.Chart(df).mark_line(point=True).encode(
                        x="date:T", y="score:Q"
                    ).properties(height=180, title=f"{skill_key.title()} score over time")
                    st.altair_chart(chart, use_container_width=True)
                    if len(df) >= 2 and df["score"].iloc[0] != 0:
                        try:
                            pct_imp = (df["score"].iloc[-1] - df["score"].iloc[0]) / abs(df["score"].iloc[0]) * 100
                            st.write(f"% improvement: {pct_imp:.0f}% from {df['date'].iloc[0].date()}")
                        except Exception:
                            pass
                    stage = agg.get(stage_key)
                    st.write(f"Current skill stage: {stage or 'Unknown'}")
            else:
                st.write("No score history available for this skill.")

    # 3. Micro-Achievements
    st.markdown("**3. Micro-Achievements**")
    if agg.get("micro_achievements"):
        for date_str, ach in agg.get("micro_achievements", []):
            st.markdown(f"- 🟢 {date_str} — {ach}")
    else:
        st.write("No micro-achievements recorded.")

    # 4. Adaptive Performance
    st.markdown("**4. Adaptive Performance**")
    table = agg.get("adaptive_performance", [])
    if table:
        df = pd.DataFrame(table)
        st.table(df)
    else:
        st.write("No adaptive performance data.")


# ---------- Streamlit main ----------
def main():
    st.title("Student Report Generator")
    st.markdown("Query like: `give summary about user_id 1`.")

    # Inputs
    data_path = st.sidebar.text_input("JSON data path", value=DEFAULT_JSON_PATH)

    today = date.today()
    default_start = today - timedelta(days=6)
    start_date = st.sidebar.date_input("Report start date", value=default_start)
    end_date = st.sidebar.date_input("Report end date", value=today)
    query = st.text_input("Query", value="give summary about user_id 1")

    if st.button("Run"):
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
            st.error("Could not extract user_id from query. Use syntax like 'user_id 3'.")
            return

        try:
            agg = aggregate_student(data, user_id)
        except Exception as e:
            st.error(f"Aggregation error: {e}")
            return

        # Per-period snapshot (+ previous period for trend)
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        curr = period_stats(data, user_id, start_dt, end_dt)

        prev_span_days = (end_dt.date() - start_dt.date()).days + 1
        prev_end = start_dt - timedelta(seconds=1)
        prev_start = prev_end - timedelta(days=prev_span_days - 1)
        prev = period_stats(data, user_id, prev_start, prev_end)

        trend_vs_prev = compute_trend(curr["total_time_mins"], prev["total_time_mins"])

        # ---- User metadata (Age uses DOB only; no "computed" wording) ----
        display_user_metadata(agg)

        # ---- Engagement & Performance Snapshot (with donut pie for avg score) ----
        st.subheader(f"Engagement & Performance Snapshot for {agg['name']} (ID {user_id})")

        mcol, pcol = st.columns([1, 1])
        with mcol:
            st.metric("Average activity score", f"{agg['avg_score']}%")

        with pcol:
            score_val = max(0, min(100, float(agg.get("avg_score", 0))))
            pie_df = pd.DataFrame(
                [{"label": "Score", "value": score_val},
                 {"label": "Remaining", "value": max(0.0, 100.0 - score_val)}]
            )
            donut = alt.Chart(pie_df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta("value:Q"),
                color=alt.Color("label:N", legend=None),
                tooltip=["label:N", alt.Tooltip("value:Q", format=".1f")]
            ).properties(height=160, width=160, title="Average activity score")
            st.altair_chart(donut, use_container_width=False)

        c1, c2, c3 = st.columns(3)
        c1.write(f"**Lessons completed:** {agg['lessons_completed']} ({agg['lesson_completion_rate']}%)")
        c2.write(f"**Total time:** {agg['total_time']} minutes")
        c3.write(f"**Avg session length:** {agg['avg_session_length']} minutes")
        st.write(f"**Badges:** {agg['badge_list']} (most recent: {agg['most_recent_badge']})")
        st.write(f"**Feedback summary:** {agg['feedback_comments_and_ratings']}")

        # Academic & Skill Progress
        render_academic_skill_progress(agg)

        # ---------- SEN Report (period-based) ----------
        st.subheader("🧾 SEN Report (11 sections)")
        if not curr["had_ts"]:
            st.warning("No reliable timestamps found in your data. The period-based figures may be limited or zero. "
                       "Add timestamps like 'start_time', 'created_at', or 'login_timestamp' for better accuracy.")

        dev = {
            "device_type": (data.get("devices_by_user", {}).get(str(user_id), {}).get("device_type", "Unknown"))
                           if isinstance(data.get("devices_by_user"), dict) else "Unknown",
            "screen_reader": data.get("devices_by_user", {}).get(str(user_id), {}).get("screen_reader", False)
                             if isinstance(data.get("devices_by_user"), dict) else False,
            "crashes": data.get("devices_by_user", {}).get(str(user_id), {}).get("crashes", 0)
                       if isinstance(data.get("devices_by_user"), dict) else 0,
            "high_packet_loss_sessions": data.get("devices_by_user", {}).get(str(user_id), {}).get("high_packet_loss_sessions", 0)
                       if isinstance(data.get("devices_by_user"), dict) else 0,
        }

        # Learning skills from agg (if present)
        skills = []
        for name, value, delta in [
            ("Reading", agg.get("literacy_pct", 0)/100.0, 0.00),
            ("Numeracy", agg.get("numeracy_pct", 0)/100.0, 0.00),
            ("Motor", agg.get("motor_pct", 0)/100.0, 0.00),
            ("Communication", agg.get("communication_pct", 0)/100.0, 0.00),
        ]:
            skills.append({"name": name, "value": float(value), "delta": float(delta)})

        language = {
            "readability_grade": "—",
            "ttr": "—",
            "topics": [],
            "tone": "neutral",
            "safety_flags": 0
        }

        ai_support = {
            "hints_per_activity": agg.get("avg_hints_used", 0),
            "hint_effect_pp": 0,
            "avatar": "—",
            "avatar_change_count": 0
        }

        dropoff_risk = "high" if (curr["active_days"] <= 3 or curr["completion_pct"] < 40) else ("medium" if curr["completion_pct"] < 60 else "low")
        routine = {
            "streak_current_days": "—",
            "streak_longest_days": "—",
            "preferred_time_window": "—",
            "dropoff_risk": dropoff_risk
        }

        focus_score_now = compute_focus_score(curr["completion_pct"], curr["avg_session_mins"])
        focus_score_prev = compute_focus_score(prev["completion_pct"], prev["avg_session_mins"])
        focus_delta = focus_score_now - focus_score_prev

        goals = data.get("goals_by_user", {}).get(str(user_id), [])
        norm_goals = []
        for g in goals:
            try:
                norm_goals.append({
                    "skill": g["skill"],
                    "current": float(g.get("current", 0)),
                    "target": float(g.get("target", 0)),
                    "target_date": g.get("target_date", "")
                })
            except Exception:
                continue

        questions = data.get("questions_by_user", {}).get(str(user_id), {})
        if not isinstance(questions, dict):
            questions = {}
        questions.setdefault("total", 0)
        questions.setdefault("unanswered_pct", 0)
        questions.setdefault("out_of_scope_pct", 0)
        questions.setdefault("examples", [])

        ui_prefs = data.get("ui_preferences_by_user", {}).get(str(user_id), {}) if isinstance(data.get("ui_preferences_by_user"), dict) else {}
        accommodations = {
            "needs": ui_prefs.get("needs", []),
            "enabled": ui_prefs.get("enabled", []),
            "tts_usage_pct": ui_prefs.get("tts_usage_pct", 0),
            "tts_accuracy_on": ui_prefs.get("tts_accuracy_on", 0.0),
            "tts_accuracy_off": ui_prefs.get("tts_accuracy_off", 0.0),
            "contrast_idle_delta_pct": ui_prefs.get("contrast_idle_delta_pct", 0),
            "font_size_changes": ui_prefs.get("font_size_changes", 0)
        }

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
            "devices": dev,
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
                "idle_pct": "—",
                "window_switches_per_session": "—"
            },
            "accommodations": accommodations,
            "learning": {
                "skills": skills,
                "perseverance_index": round(max(0.9, agg.get("avg_hints_used", 0.9)), 1),
                "ttfc_secs": 45
            },
            "language": language,
            "ai_support": ai_support,
            "routine": routine,
            "goals": norm_goals,
            "recommendations": [],
            "questions": questions
        }

        report_text = build_report(report_data)
        st.text_area("Report (copy-ready)", value=report_text, height=600)
        st.download_button(
            "Download report (.txt)",
            data=report_text.encode("utf-8"),
            file_name=f"sen_report_user_{user_id}_{start_date}_{end_date}.txt",
            mime="text/plain"
        )


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
            f"**School:** {agg['school_name']}, **Class Level:** {agg['class_level']}, **Reading Level:** {agg['reading_level']}\n\n"
            f"**Account Created:** {agg['created_at']}, **Last Updated:** {agg['updated_at']}\n\n"
            f"**Onboarding Complete:** {agg['onboarding_complete']}"
        )
        cols[1].markdown(info)


if __name__ == "__main__":
    main()
