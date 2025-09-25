# ui_personalisation.py
import streamlit as st
import pandas as pd
from typing import Dict, Optional
from constants import AVATAR_KEYS, FONT_KEYS, BACKGROUND_KEYS
from utils_time import pick_first_ts, get_time_spent
from joins import build_indexes, topic_session_user_id, chapter_session_user_id, perf_user_id
from charts_altair import meter_chart
import altair as alt

def _extract_attr(record: Dict, candidates):
    for k in candidates:
        if k in record and record[k] not in (None, ""):
            return str(record[k])
    for key in record.keys():
        lk = key.lower()
        for c in candidates:
            if c.lower() in lk and record.get(key) not in (None, ""):
                return str(record[key])
    return None

def collect_personalisation_usage(data: Dict, user_id: int,
                                  start_dt: Optional[pd.Timestamp],
                                  end_dt: Optional[pd.Timestamp]) -> Dict[str, Dict[str, float]]:
    idx = build_indexes(data)
    usage = {"avatar": {}, "font": {}, "background": {}}

    def _within(r, keys):
        if not start_dt or not end_dt:
            return True
        ts = pick_first_ts(r, keys)
        return (ts is not None) and (start_dt <= ts <= end_dt)

    def _add(kind, label, minutes):
        if not label or minutes <= 0:
            return
        usage[kind][label] = usage[kind].get(label, 0.0) + minutes

    for r in data.get("topic_session", []):
        if topic_session_user_id(r, idx) != user_id: continue
        if not _within(r, ["started_at", "completed_at", "created_at", "timestamp", "date"]): continue
        mins = get_time_spent(r)
        _add("avatar", _extract_attr(r, AVATAR_KEYS), mins)
        _add("font", _extract_attr(r, FONT_KEYS), mins)
        _add("background", _extract_attr(r, BACKGROUND_KEYS), mins)

    for r in data.get("chapter_session", []):
        if chapter_session_user_id(r, idx) != user_id: continue
        if not _within(r, ["started_at", "completed_at", "created_at", "timestamp", "date"]): continue
        mins = get_time_spent(r)
        _add("avatar", _extract_attr(r, AVATAR_KEYS), mins)
        _add("font", _extract_attr(r, FONT_KEYS), mins)
        _add("background", _extract_attr(r, BACKGROUND_KEYS), mins)

    for r in data.get("lesson_session", []):
        if r.get("user_id") != user_id: continue
        if not _within(r, ["created_at", "started_at", "completed_at", "timestamp", "date"]): continue
        mins = get_time_spent(r)
        _add("avatar", _extract_attr(r, AVATAR_KEYS), mins)
        _add("font", _extract_attr(r, FONT_KEYS), mins)
        _add("background", _extract_attr(r, BACKGROUND_KEYS), mins)

    for r in data.get("daily_activity_log", []):
        if r.get("user_id") != user_id: continue
        if not _within(r, ["login_timestamp", "created_at", "timestamp", "date"]): continue
        mins = get_time_spent(r)
        _add("avatar", _extract_attr(r, AVATAR_KEYS), mins)
        _add("font", _extract_attr(r, FONT_KEYS), mins)
        _add("background", _extract_attr(r, BACKGROUND_KEYS), mins)

    for r in data.get("activity_performance", []):
        if perf_user_id(r, idx) != user_id: continue
        if not _within(r, ["submitted_at"]): continue
        mins = get_time_spent(r)
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

def render_personalisation_usage(data, user_id, start_dt, end_dt):
    st.subheader("🎭 Personalisation usage (Avatar • Font • Background)")
    tabs = st.tabs(["This period", "All time"])
    with tabs[0]:
        usage_p = collect_personalisation_usage(data, user_id, start_dt, end_dt)
        df_p = _usage_df(usage_p)
        c1, c2, c3 = st.columns(3)
        with c1: st.altair_chart(_usage_chart(df_p, "avatar", "period"), use_container_width=True)
        with c2: st.altair_chart(_usage_chart(df_p, "font", "period"), use_container_width=True)
        with c3: st.altair_chart(_usage_chart(df_p, "background", "period"), use_container_width=True)
        if not df_p.empty:
            st.download_button("Download period personalisation (.csv)",
                               df_p.to_csv(index=False).encode("utf-8"),
                               file_name=f"user_{user_id}_personalisation_period.csv",
                               mime="text/csv")
        else:
            st.info("No personalisation usage found in the selected period.")
    with tabs[1]:
        usage_a = collect_personalisation_usage(data, user_id, None, None)
        df_a = _usage_df(usage_a)
        c1, c2, c3 = st.columns(3)
        with c1: st.altair_chart(_usage_chart(df_a, "avatar", "all-time"), use_container_width=True)
        with c2: st.altair_chart(_usage_chart(df_a, "font", "all-time"), use_container_width=True)
        with c3: st.altair_chart(_usage_chart(df_a, "background", "all-time"), use_container_width=True)
        if not df_a.empty:
            st.download_button("Download all-time personalisation (.csv)",
                               df_a.to_csv(index=False).encode("utf-8"),
                               file_name=f"user_{user_id}_personalisation_alltime.csv",
                               mime="text/csv")
        else:
            st.info("No personalisation usage found in the dataset.")
