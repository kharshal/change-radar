# ============================================================================
# Project Structure:
# 
# change_radar/
# ├── app.py                    # Main entry point (THIS FILE)
# ├── config/
# │   ├── __init__.py
# │   └── settings.py           # Configuration constants
# ├── components/
# │   ├── __init__.py
# │   ├── header.py             # Header component
# │   ├── feedback_modal.py     # Feedback dialog
# │   └── kpi_card.py           # Reusable KPI card
# ├── pages/
# │   ├── __init__.py
# │   ├── build_knowledge.py       # Data upload and setup
# │   ├── dashboard.py          # Main insights dashboard
# │   └── deep_dive.py          # Deep dive analysis
# ├── services/
# │   ├── __init__.py
# │   ├── data_service.py       # Data generation and processing
# │   └── kpi_service.py        # KPI extraction and management
# ├── utils/
# │   ├── __init__.py
# │   ├── session_state.py      # Session state management
# │   └── styles.py             # CSS styles
# └── requirements.txt
# ============================================================================


"""
Main application entry point.
"""
import streamlit as st

# MUST BE FIRST
st.set_page_config(
    page_title="Change Radar",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Now import everything else with updated names
import sys
import importlib

from config.config import Config  # UPDATED
from utils.session_state import SessionState
from utils.styles import get_custom_css
from components.header import render_header
from pages.build_knowledge import BuildKnowledge  # UPDATED
from pages.deep_dive import DeepDive
from pages.dashboard import Dashboard


# Configuration
DEV_MODE = True

def reload_all_modules():
    """Reload all custom modules during development."""
    if not DEV_MODE:
        return
    
    custom_modules = [
        'config.config',  # UPDATED
        'utils.session_state',
        'utils.styles',
        'services.data_service',
        'services.kpi_service',
        'services.file_service',  # NEW
        'services.generation_service',  # NEW
        'components.header',
        'components.feedback_modal',
        'components.kpi_card',
        'pages.build_knowledge',  # UPDATED
        'pages.dashboard',
        'pages.deep_dive',
    ]
    
    for module_name in custom_modules:
        if module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
            except Exception as e:
                pass

if DEV_MODE:
    reload_all_modules()

def main():
    """Main application function."""
    SessionState.initialize()
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    render_header()
    
    if DEV_MODE:
        st.sidebar.warning("🔧 Dev Mode")
        if st.sidebar.button("🔄 Reload"):
            reload_all_modules()
            st.rerun()
    
    if SessionState.get('show_deep_dive'):
#         from pages.deep_dive import DeepDive
        DeepDive.render()
    elif SessionState.get('show_dashboard'):
#         from pages.dashboard import Dashboard
        Dashboard.render()
    else:
        left, center, right = st.columns([1, 2, 1])
        with center:
            BuildKnowledge.render()  # UPDATED


if __name__ == "__main__":
    main()                        









# # ============================================================================
# # FILE: app.py (Main Entry Point)
# # ============================================================================
# """
# Main application entry point.
# """
# import streamlit as st
# from config.config import Config
# from utils.session_state import SessionState
# from utils.styles import get_custom_css
# from components.header import render_header
# from pages.build_knowledge import BuildKnowledge

# def main():
#     """Main application function."""
#     # Page configuration
#     st.set_page_config(
#         page_title=Config.APP_TITLE,
#         page_icon=Config.APP_ICON,
#         layout=Config.PAGE_LAYOUT,
#         initial_sidebar_state="collapsed"
#     )
        
#     # Initialize session state
#     SessionState.initialize()
    
#     # Apply custom styles
#     st.markdown(get_custom_css(), unsafe_allow_html=True)
    
#     # Render header
#     render_header()
    
#     # Route to appropriate page - IMPORTANT: Check deep_dive FIRST
#     if SessionState.get('show_deep_dive'):
#         # Import and render deep dive
#         from pages.deep_dive import DeepDive
#         DeepDive.render()
#     elif SessionState.get('show_dashboard'):
#         # Import and render dashboard
#         from pages.dashboard import Dashboard
#         Dashboard.render()
#     else:
#         # Render setup wizard
#         left, center, right = st.columns([1, 2, 1])
#         with center:
#             SetupWizard.render()
    
# if __name__ == "__main__":
#     main()

       