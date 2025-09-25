# ui_subjects.py
import streamlit as st
import pandas as pd
import altair as alt

def render_subject_growth(agg):
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
