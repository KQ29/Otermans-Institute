# ui_profile.py
import streamlit as st

def display_user_metadata(agg):
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
