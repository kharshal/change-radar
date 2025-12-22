# ============================================================================
# FILE: utils/session_state.py
# ============================================================================
"""
Session state management utilities.
"""
import streamlit as st
from typing import Any, List, Dict


class SessionState:
    """Manage Streamlit session state."""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables."""
        defaults = {
            'session_folder': None,
            'data_files': [],
            'dkl_output': None,
            'dkl_generated': False,
            'fkl_files': [],          
            'fkl_uploaded': False,
            'causal_graph_generated': False,
            'kpi_list': [],
            'selected_kpis': [],
            'active_kpi': None,
            'show_dashboard': False,
            'show_deep_dive': False,
            'deep_dive_driver': None,
            'show_feedback': False,
            'feedback_data': [],
            'table_descriptions': {}, #REMOVE
            'warehouse_info': {} #REMOVE
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get a session state value."""
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key: str, value: Any):
        """Set a session state value."""
        st.session_state[key] = value
    
    @staticmethod
    def append(key: str, value: Any):
        """Append to a list in session state."""
        if key not in st.session_state:
            st.session_state[key] = []
        st.session_state[key].append(value)
            