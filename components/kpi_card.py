# ============================================================================
# FILE: components/kpi_card.py
# ============================================================================
"""
Reusable KPI card component.
"""
import streamlit as st


def render_kpi_card(kpi: str, data: dict, is_active: bool, button_key: str):
    """
    Render a single KPI selection card.
    
        Args:
        kpi: KPI name
        data: KPI data dictionary with 'change_pct' key
        is_active: Whether this KPI is currently active
        button_key: Unique button key
    """
    change_pct = data['change_pct']
    is_positive = change_pct > 0
    
    dot = "🟢" if is_positive else "🔴"
    sign = "+" if is_positive else "-"
    button_label = f"{kpi}  | {dot} {sign}{abs(change_pct):.1f}%"
    
    if st.button(
        button_label,
        key=button_key,
        use_container_width=True,
        help=f"Click to analyze {kpi}"
    ):
        st.session_state.active_kpi = kpi
        st.rerun()

        