# ============================================================================
# FILE: components/feedback_modal.py
# ============================================================================
"""
Feedback modal component.
"""
import streamlit as st
import pandas as pd
from utils.session_state import SessionState


@st.dialog("💬 Give Feedback")
def show_feedback_modal():
    """Display feedback collection modal."""
    st.markdown("#### Do you agree with the Analysis?")
    
    agree_choice = st.radio(
        "",
        options=["Yes", "No"],
        key="feedback_agree",
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("<div style='margin: 16px 0;'></div>", unsafe_allow_html=True)
    
    feedback_text = ""
    if agree_choice == "No":
        st.markdown("**Please provide additional comments:**")
        feedback_text = st.text_area(
            "",
            placeholder="Share your thoughts here...",
            height=120,
            key="feedback_text",
            label_visibility="collapsed"
        )
    
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    
    if st.button("Submit Feedback", type="primary", use_container_width=True):
        feedback_entry = {
            "timestamp": pd.Timestamp.now(),
            "kpi": SessionState.get('active_kpi'),
            "agreement": agree_choice,
            "comments": feedback_text if agree_choice == "No" else ""
        }
        
        SessionState.append('feedback_data', feedback_entry)
        st.success("✓ Feedback submitted successfully!")
        st.balloons()
        st.rerun()
