# ============================================================================
# FILE: services/data_service.py
# ============================================================================
"""
Data generation and processing services.
"""
import random
import pandas as pd
from typing import Dict, List
from config.config import Config


class DataService:
    """Handle data generation and processing."""
    
    @staticmethod
    def generate_mock_timeseries(kpi_name: str, seed: int = 42) -> Dict:
        """Generate mock time series data for a KPI."""
        random.seed(hash(kpi_name + str(seed)))
        
        base_value = random.randint(3000, 5000)
        values = [base_value + random.randint(-500, 800) for _ in range(12)]

        mean_value = pd.Series(values).mean()
        change_pct = (values[-1] - mean_value) / mean_value * 100
        
        change_pct_last = (values[-1] - values[-2]) / values[-2] * 100
        print(kpi_name, mean_value, change_pct, change_pct_last)
                
        return {
            'months': Config.MONTHS,
            'values': values,
            'current_value': values[-1],
            'change_pct': change_pct,
            'change_pct_last': change_pct_last
        }
    
    @staticmethod
    def generate_breakdown_data(dimension: str, kpi_name: str) -> pd.DataFrame:
        """Generate breakdown data by dimension."""
        dimension_map = {
            "Country": ["USA", "India", "Germany", "Japan", "UK", "France", "Canada", "Australia"],
            "Channel": ["E-commerce", "Retail", "Wholesale", "Direct", "Partner", "Marketplace"],
            "Region": ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
        }
        
        items = dimension_map.get(dimension, ["Category A", "Category B", "Category C"])
        
        data = []
        for item in items:
            data.append({
                "FACTOR": item,
                "VALUE": f"{random.uniform(10, 50):.1f}M",
                "% CHANGE": random.randint(-10, 20),
                "% CONTRIB": f"{random.randint(5, 40)}%"
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_causal_drivers(kpi_name: str) -> List[Dict]:
        """Generate causal drivers for a KPI."""
        return [
            {
                "title": "Increase in Number of Order",
                "impact": random.randint(85, 98),
                "description": f"Seasonal increase in demand during Q4 contributed to {kpi_name.lower()} growth."
            }
        ]
