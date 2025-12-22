# ============================================================================
# FILE: services/kpi_service.py
# ============================================================================
"""
KPI extraction and management services.
"""
import pandas as pd
from typing import List
import streamlit as st


class KPIService:
    """Handle KPI extraction and management."""
    
    @staticmethod
    def extract_from_excel(files) -> List[str]:
        """Extract KPI names from uploaded Excel files."""
        kpi_names = []
        kpi_columns = ['KPI Name', 'kpi name', 'KPI_Name', 'kpi_name']
        
        for file in files:
            try:
                if file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file)
                    
                    for col in kpi_columns:
                        if col in df.columns:
                            kpi_names.extend(df[col].dropna().unique().tolist())
                            break
                            
            except Exception as e:
                st.warning(f"Could not extract KPIs from {file.name}: {str(e)}")
        
        return list(set(kpi_names))
    
    
    
