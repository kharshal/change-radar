# ============================================================================
# FILE: pages/deep_dive.py
# ============================================================================
"""
Deep dive analysis page for causal drivers.
"""
import streamlit as st
import plotly.graph_objects as go
import random
from typing import Dict

from utils.session_state import SessionState
from config.config import Config


class DeepDive:
    """Deep dive analysis page."""
    
    @staticmethod
    def render():
        """Render the deep dive page."""
        # Header
        DeepDive._render_header()
        
        # Main layout
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            DeepDive._render_relationship_graph()
        
        with col_right:
            DeepDive._render_impact_analysis()
        
        st.stop()
    
    @staticmethod
    def _render_header():
        """Render page header with back button."""
        col1, col2, _ = st.columns([1, 1, 1])
        
        with col1:
            if st.button("← Back"):
                # Reset deep dive state
                SessionState.set('show_deep_dive', False)
                SessionState.set('deep_dive_driver', None)
                # Keep dashboard open
                SessionState.set('show_dashboard', True)
                st.rerun()
        
        with col2:
            driver = SessionState.get('deep_dive_driver')
            if driver:
                st.title(f"KPI Deep Dive: {driver['title']}")
        
        st.markdown("---")
    
    
    @staticmethod
    def _render_relationship_graph():
        """Render causal relationship graph."""
        st.subheader("Relationship Graph")
        
        fig = DeepDive._create_causal_graph()
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _create_causal_graph() -> go.Figure:
        """Create causal relationship graph visualization."""
        driver = SessionState.get('deep_dive_driver')
        active_kpi = SessionState.get('active_kpi')
        
        fig = go.Figure()
        
        # Define nodes and positions
        nodes = [
            "Marketing Spend",
            "Customer Traffic",
            active_kpi,
            driver['title'],
            "Conversion Rate"
        ]
        x_pos = [0.2, 0.5, 0.8, 0.3, 0.7]
        y_pos = [0.8, 0.5, 0.5, 0.2, 0.2]
        
        # Define edges
        edges = [(0, 1), (0, 3), (3, 4), (4, 1), (1, 2)]
        
        # Add edges as lines
        for edge in edges:
            fig.add_trace(go.Scatter(
                x=[x_pos[edge[0]], x_pos[edge[1]]],
                y=[y_pos[edge[0]], y_pos[edge[1]]],
                mode='lines',
                line=dict(color=Config.COLORS['primary'], width=3),
                showlegend=False,
                hoverinfo='none'
            ))
        
        # Add arrow annotations
        for edge in edges:
            fig.add_annotation(
                x=x_pos[edge[1]],
                y=y_pos[edge[1]],
                ax=x_pos[edge[0]],
                ay=y_pos[edge[0]],
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=Config.COLORS['primary']
            )
        
        # Add nodes
        for i, node in enumerate(nodes):
            if node == driver['title']:
                color = '#7c3aed'
            elif node == active_kpi:
                color = Config.COLORS['danger']
            else:
                color = Config.COLORS['primary']
            
            fig.add_trace(go.Scatter(
                x=[x_pos[i]],
                y=[y_pos[i]],
                mode='markers+text',
                marker=dict(size=50, color=color, line=dict(color='white', width=2)),
                text=[node],
                textposition='bottom center',
                textfont=dict(size=11, color='#1f2937', family='Arial Black'),
                showlegend=False,
                hoverinfo='text',
                hovertext=node
            ))
        
        fig.update_layout(
            height=500,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.1, 1.1]),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.1, 1.1]),
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        return fig
    
    @staticmethod
    def _render_impact_analysis():
        """Render detailed impact analysis."""
        driver = SessionState.get('deep_dive_driver')
        active_kpi = SessionState.get('active_kpi')
        
        if not driver or not active_kpi:
            return
        
        # Generate random statistics
        correlation = random.randint(75, 95)
        time_lag = random.randint(3, 14)
        traffic_increase = random.randint(10, 25)
        conversion_improve = random.randint(5, 15)
        transaction_change = random.randint(-5, 20)
        variance_explained = random.randint(40, 70)
        peak_lag = random.randint(1, 7)
        
        st.markdown(f"""
        #### Impact Analysis: {driver['title']}
        
        **Overall Impact**: {driver['impact']}% contribution to {active_kpi}
        
        {driver['description']}
        
        #### Detailed Causal Chain
        
        ** Primary Driver**
        - {driver['title']} showed significant variation in the analyzed period
        - Direct correlation coefficient with {active_kpi}: 0.{correlation}
        - Time lag effect: {time_lag} days
        
        ** Intermediate Effects**
        - Customer traffic increased by {traffic_increase}%
        - Conversion rate improved by {conversion_improve}%
        - Average transaction value changed by {transaction_change}%
        
        """)
