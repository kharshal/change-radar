# ============================================================================
# FILE: components/header.py
# ============================================================================
"""
Header component.
"""
import streamlit as st


def render_header(title: str = "Change Radar"):
    """Render application header."""
    st.columns([1, 6])[0].markdown(
        f"""
        <div style="font-size:16px; font-weight:600; color:#111827;">
            🧠 {title}
        </div>
        """,
        unsafe_allow_html=True
    )
