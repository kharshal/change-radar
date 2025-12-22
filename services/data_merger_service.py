# ============================================================================
# FILE: services/data_merger_service.py (NEW)
# ============================================================================
"""
Service for merging uploaded data files and preparing for analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class DataMergerService:
    """Handle data file merging and KPI discovery."""
    
    @staticmethod
    def load_files(data_folder: str) -> Dict[str, pd.DataFrame]:
        """
        Load all data files from folder.
        
        Args:
            data_folder: Path to data folder
            
        Returns:
            Dict mapping filename to DataFrame
        """
        data_path = Path(data_folder)
        dataframes = {}
        
        for file_path in data_path.glob('*'):
            if file_path.suffix in ['.csv', '.xlsx', '.xls']:
                try:
                    if file_path.suffix == '.csv':
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_excel(file_path)
                    
                    dataframes[file_path.name] = df
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
        
        return dataframes
    
    @staticmethod
    def parse_dkl_config(dkl_config: Dict) -> Dict:
        """
        Parse DKL configuration to extract merge instructions.
        
        Expected DKL structure:
        {
            "primary_table": "orders.csv",
            "primary_key": "order_id",
            "date_column": "order_date",
            "merge_instructions": [
                {"table": "payments.csv", "on": "order_id", "how": "left"},
                {"table": "customers.csv", "on": "customer_id", "how": "left"}
            ],
            "aggregation_level": "daily",  # or "weekly", "monthly"
            "kpi_columns": ["order_id", "payment_value"]
        }
        
        Args:
            dkl_config: DKL configuration dictionary
            
        Returns:
            Parsed configuration
        """
        return {
            'primary_table': dkl_config.get('primary_table'),
            'primary_key': dkl_config.get('primary_key'),
            'date_column': dkl_config.get('date_column'),
            'merge_instructions': dkl_config.get('merge_instructions', []),
            'aggregation_level': dkl_config.get('aggregation_level', 'daily'),
            'kpi_columns': dkl_config.get('kpi_columns', [])
        }
    
    @staticmethod
    def auto_detect_date_column(df: pd.DataFrame) -> Optional[str]:
        """
        Automatically detect date column in DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of date column or None
        """
        date_keywords = ['date', 'time', 'timestamp', 'day', 'month', 'year']
        
        for col in df.columns:
            col_lower = col.lower()
            # Check column name
            if any(keyword in col_lower for keyword in date_keywords):
                try:
                    pd.to_datetime(df[col])
                    return col
                except:
                    continue
        
        # Check data types
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                return col
        
        return None
    
    @staticmethod
    def infer_granularity(df: pd.DataFrame, date_col: str) -> str:
        """
        Infer time granularity from date column.
        
        Args:
            df: DataFrame with date column
            date_col: Name of date column
            
        Returns:
            'daily', 'weekly', or 'monthly'
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Calculate median difference between consecutive dates
        sorted_dates = df[date_col].sort_values().unique().dropna()
        if len(sorted_dates) < 2:
            return 'daily'
        
        diffs = sorted_dates.diff().dropna()
        median_diff = diffs.median().days
        
        if median_diff <= 1:
            return 'daily'
        elif median_diff <= 7:
            return 'weekly'
        else:
            return 'monthly'
    
    @staticmethod
    def merge_dataframes(
        dataframes: Dict[str, pd.DataFrame],
        dkl_config: Dict
    ) -> pd.DataFrame:
        """
        Merge multiple dataframes based on DKL configuration.
        
        Args:
            dataframes: Dictionary of loaded dataframes
            dkl_config: DKL configuration
            
        Returns:
            Merged DataFrame
        """
        config = DataMergerService.parse_dkl_config(dkl_config)
        
        # Start with primary table
        primary_table = config['primary_table']
        if primary_table not in dataframes:
            raise ValueError(f"Primary table {primary_table} not found in data files")
        
        merged_df = dataframes[primary_table].copy()
        
        # Merge additional tables
        for merge_instr in config['merge_instructions']:
            table_name = merge_instr['table']
            if table_name not in dataframes:
                print(f"Warning: {table_name} not found, skipping merge")
                continue
            
            merged_df = pd.merge(
                merged_df,
                dataframes[table_name],
                on=merge_instr['on'],
                how=merge_instr.get('how', 'left')
            )
        
        return merged_df
    
    @staticmethod
    def discover_kpis(df: pd.DataFrame, date_col: str) -> List[Dict]:
        """
        Automatically discover potential KPIs from data.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            
        Returns:
            List of discovered KPIs with metadata
        """
        discovered_kpis = []
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col == date_col:
                continue
            
            kpi_info = {
                'name': col,
                'type': 'numeric',
                'aggregation': 'sum',  # default
                'description': f'Aggregated {col}',
                'sample_values': df[col].dropna().head(5).tolist()
            }
            
            # Determine appropriate aggregation
            if 'count' in col.lower() or 'num' in col.lower() or 'quantity' in col.lower():
                kpi_info['aggregation'] = 'sum'
            elif 'avg' in col.lower() or 'mean' in col.lower() or 'average' in col.lower():
                kpi_info['aggregation'] = 'mean'
            elif 'price' in col.lower() or 'amount' in col.lower() or 'value' in col.lower():
                kpi_info['aggregation'] = 'sum'
            
            discovered_kpis.append(kpi_info)
        
        # Count-based KPIs (row counts)
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        for id_col in id_columns:
            kpi_info = {
                'name': f'total_{id_col.replace("_id", "s")}',
                'type': 'count',
                'aggregation': 'count',
                'description': f'Count of unique {id_col}',
                'base_column': id_col
            }
            discovered_kpis.append(kpi_info)
        
        return discovered_kpis
    
    @staticmethod
    def aggregate_data(
        df: pd.DataFrame,
        date_col: str,
        kpis: List[Dict],
        granularity: str = 'daily'
    ) -> pd.DataFrame:
        """
        Aggregate data to specified granularity level.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            kpis: List of KPI definitions
            granularity: 'daily', 'weekly', or 'monthly'
            
        Returns:
            Aggregated DataFrame
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Determine grouping frequency
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'ME'
        }
        freq = freq_map.get(granularity, 'D')
        
        # Create date grouping column
        if granularity == 'daily':
            df['agg_date'] = df[date_col].dt.date
        elif granularity == 'weekly':
            df['agg_date'] = df[date_col] - pd.to_timedelta(df[date_col].dt.dayofweek, unit='d')
        else:  # monthly
            df['agg_date'] = df[date_col].dt.to_period('M').dt.to_timestamp()
        
        # Build aggregation dictionary
        agg_dict = {}
        for kpi in kpis:
            if kpi['type'] == 'count':
                agg_dict[kpi['name']] = (kpi['base_column'], 'count')
            else:
                agg_method = kpi.get('aggregation', 'sum')
                agg_dict[kpi['name']] = (kpi['name'], agg_method)
        
        # Perform aggregation
        aggregated = df.groupby('agg_date').agg(**agg_dict).reset_index()
        aggregated.rename(columns={'agg_date': date_col}, inplace=True)
        
        return aggregated


