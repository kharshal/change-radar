# ============================================================================
# FILE: pages/dashboard.py
# ============================================================================
"""
Main insights dashboard page.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List

from utils.session_state import SessionState
from services.data_service import DataService
from components.feedback_modal import show_feedback_modal
from components.kpi_card import render_kpi_card
from config.config import Config


class Dashboard:
    """Main insights dashboard."""
    
    @staticmethod
    def render():
        """Render the complete dashboard."""
        # Header with back button
        Dashboard._render_header()
        
        # Initialize active KPI
        Dashboard._initialize_active_kpi()
        
        # Generate data for all selected KPIs
        kpi_data = Dashboard._generate_kpi_data()
        
        # Main layout
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            Dashboard._render_kpi_selector(kpi_data)
        
        with col_right:
            Dashboard._render_chart_section(kpi_data)
        
        # Narrative section
        Dashboard._render_narrative_section(kpi_data)
        
        # Bottom sections
        col_left_bottom, col_right_bottom = st.columns([1, 1])
        
        with col_left_bottom:
            Dashboard._render_impact_factors()
        
        with col_right_bottom:
            Dashboard._render_causal_drivers()
        
        st.stop()
    
    @staticmethod
    def _render_header():
        """Render dashboard header with back button."""
        col1, col2, _ = st.columns([1, 1, 1])
        
        with col1:
            if st.button("← Back"):
                SessionState.set('show_dashboard', False)
                SessionState.set('active_kpi', None)
                st.rerun()
        
        with col2:
            st.title("Insights Dashboard")
        
        st.markdown("---")
    
    @staticmethod
    def _initialize_active_kpi():
        """Set first KPI as active if not set."""
        if SessionState.get('active_kpi') is None and SessionState.get('selected_kpis'):
            SessionState.set('active_kpi', SessionState.get('selected_kpis')[0])
    
    @staticmethod
    def _generate_kpi_data() -> Dict:
        """Generate mock data for all selected KPIs."""
        kpi_data = {}
        for kpi in SessionState.get('selected_kpis', []):
            kpi_data[kpi] = DataService.generate_mock_timeseries(kpi, seed=42)
        return kpi_data
    
    @staticmethod
    def _render_kpi_selector(kpi_data: Dict):
        """Render KPI selection panel."""
        st.markdown(
            """
            <div style="
                font-size:24px; font-weight:600;
                color:#1f2937; margin-bottom:0px;
                padding:12px 16px;
                border-radius:6px; background-color:#BFDBFE;
            ">
                📊 Select KPI to monitor
            </div>
            """,
            unsafe_allow_html=True
        )
        
        with st.container(height=435):
            for i, kpi in enumerate(SessionState.get('selected_kpis', [])):
                data = kpi_data[kpi]
                change_pct = data['change_pct']
                is_positive = change_pct > 0
                
                dot = "🟢" if is_positive else "🔴"
                sign = "+" if is_positive else ""
                button_label = f"{kpi}  | {dot} {sign}{abs(change_pct):.1f}%"
                button_key = f"select_kpi_{i}_{kpi}"
                
                if st.button(
                    button_label,
                    key=button_key,
                    use_container_width=True,
                    help=f"Click to analyze {kpi}"
                ):
                    SessionState.set('active_kpi', kpi)
                    st.rerun()
    
    @staticmethod
    def _render_chart_section(kpi_data: Dict):
        """Render chart and feedback section."""
        with st.container(height=500):
            active_kpi = SessionState.get('active_kpi')
            
            if active_kpi:
                active_data = kpi_data[active_kpi]
                
                # Chart header
                st.markdown(
                    f"""
                    <div style="border:1px solid #ccc; padding:12px 16px; 
                        border-radius:6px; background-color:#F3F4F6;
                        color:#111827; margin: -8px 0 8px 0;">
                        <div style="font-size:16px; font-weight:600;">📈 {active_kpi} Chart</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Create and display chart
                fig = Dashboard._create_line_chart(active_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feedback button
                Dashboard._render_feedback_button()
    
    @staticmethod
    def _create_line_chart(data: Dict) -> go.Figure:
        """Create line chart for KPI trend."""
        fig = go.Figure()
        
        # Color last point differently
        colors = ['#3b82f6'] * 11 + ['#ef4444']
        
        fig.add_trace(go.Scatter(
            x=data['months'],
            y=data['values'],
            mode='lines+markers',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8, color=colors, line=dict(color='white', width=2)),
            hovertemplate='<b>%{x}</b><br>Value: %{y:,.0f}<extra></extra>'
        ))
        
        # Add average line
        avg_value = sum(data['values']) / len(data['values'])
        fig.add_hline(
            y=avg_value,
            line_dash="dot",
            line_color="#22c55e",
            annotation_text=f"Avg: {avg_value:,.0f}",
            annotation_position="right"
        )
        
        fig.update_layout(
            height=Config.CHART_HEIGHT,
            xaxis_title="",
            yaxis_title="",
            plot_bgcolor='white',
            hovermode='x unified',
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis=dict(showgrid=True, gridcolor='#f3f4f6'),
            yaxis=dict(showgrid=True, gridcolor='#f3f4f6')
        )
        
        return fig
    
    @staticmethod
    def _render_feedback_button():
        """Render feedback button."""
        col_fb1, col_fb2, col_fb3 = st.columns([3, 1, 1])
        
        with col_fb3:
            if st.button("💬 Give Feedback", use_container_width=True):
                show_feedback_modal()
    
    @staticmethod
    def _render_narrative_section(kpi_data: Dict):
        """Render KPI insight narrative."""
        active_kpi = SessionState.get('active_kpi')
        
        if active_kpi:
            active_data = kpi_data[active_kpi]
            show_alert = (abs(active_data["change_pct_last"]) > 5) and (abs(active_data["change_pct"]) > 12)
            
            direction = "above" if active_data['change_pct'] > 0 else "below"
            abs_change = abs(active_data['change_pct'])
            
            narrative = f"""{active_kpi} for Nov 2025, is {abs_change:.1f}% {direction} the yearly average between Nov 2024 and Oct 2025. Value of {active_data['current_value']:,} reported for Nov 2025, is 220% higher than what was reported during Nov 2024, and 70% higher than the reported {active_kpi} in Oct 2025."""
            
            # Generate alert badge if needed
            alert_badge = ""
            if show_alert:
                alert_badge = '''
                <span style="display: inline-block; background: #fee2e2;
                    color: #dc2626; padding: 4px 12px; border-radius: 16px;
                    font-size: 11px; font-weight: 600; margin-left: 10px; vertical-align: middle;
                "> 🔴 Abnormal </span>
                '''

            # Render the insight box
            st.markdown(
                f"""
                <div class="narrative-box">
                    <div style="display: flex; align-items: center; margin: 0;">
                        <h4 style="margin: 0; font-size: 16px; font-weight: 600; color: #1f2937;">
                            💡 KPI Insight and Movement Analysis
                        </h4>
                        {alert_badge}
                </div>
                <p >{narrative}</p>
                """,
                unsafe_allow_html=True
            )
                        
            
    @staticmethod
    def _render_impact_factors():
        """Render impact factors breakdown table."""
        st.markdown(
            f"""
            <div style="
                font-size:24px; font-weight:600;
                color:#1f2937; margin-bottom:0px;
                padding:12px 16px;
                border-radius:6px; background-color:#BFDBFE;
            ">🧩 Where all {SessionState.get('active_kpi')} has changed
            </div>
            """,
            unsafe_allow_html=True
        )
        
        with st.container(height=500):
            # Dimension selector
            dimension = st.selectbox(
                "Select Dimension",
                ["Country", "Channel", "Region"],
                key="dimension_selector"
            )
            
            active_kpi = SessionState.get('active_kpi')
            
            if active_kpi:
                # Generate and display breakdown table
                df_breakdown = DataService.generate_breakdown_data(dimension, active_kpi)
                
                # Style the dataframe
                def color_change(val):
                    if isinstance(val, (int, float)):
                        color = Config.COLORS['success'] if val > 0 else Config.COLORS['danger']
                        return f'color: {color}; font-weight: 600;'
                    return ''
                
                styled_df = df_breakdown.style.applymap(color_change, subset=['% CHANGE'])
                
                st.dataframe(
                    styled_df.set_properties(**{
                        "white-space": "nowrap",
                        "overflow": "hidden",
                        "text-overflow": "ellipsis"
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
    
    @staticmethod
    def _render_causal_drivers():
        """Render causal drivers section."""
        st.markdown(
            f"""
            <div style="
                font-size:24px; font-weight:600;
                color:#1f2937; margin-bottom:0px;
                padding:12px 16px;
                border-radius:6px; background-color:#BFDBFE;
            ">🔍 Why {SessionState.get('active_kpi')} has changed
            </div>
            """,
            unsafe_allow_html=True
        )
        
        with st.container(height=500):
            active_kpi = SessionState.get('active_kpi')
            
            print(active_kpi)
            if active_kpi:
                
                
                if active_kpi == 'Total Payments':
                    drivers = DataService.generate_causal_drivers(active_kpi)
                else:
                    drivers = {}
                
                for i, driver in enumerate(drivers):
                    col_d1, col_d2 = st.columns([4, 1])
                    
                    with col_d1:
                        st.markdown(
                            f"""
                            <div class="driver-card" style="
                                margin:-10px -10px -20px 0px;
                                min-height: 80px;
                                display: flex;
                                align-items: center;
                            ">
                                <div class="driver-desc">{driver['description']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    with col_d2:
                        st.markdown(
                            '<div style="margin:-70px 0px -20px -10px; min-height: 80px; display: flex; align-items: center;">',
                            unsafe_allow_html=True
                        )
                        if st.button("Show More →", key=f"driver_{i}", use_container_width=True):
                            SessionState.set('show_deep_dive', True)
                            SessionState.set('deep_dive_driver', driver)
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)
