# ============================================================================
# FILE: config/config.py
# ============================================================================
"""
Application configuration and constants.
"""

class Config:
    """Application configuration."""
    
    # App metadata
    APP_TITLE = "Change Radar"
    APP_ICON = "🧠"
    PAGE_LAYOUT = "wide"
    
    # File upload settings
    ALLOWED_DATA_TYPES = ['csv', 'xlsx', 'json']
    ALLOWED_DOC_TYPES = ['docx', 'pdf', 'txt', 'csv', 'xlsx', 'json']
    
    # Chart settings
    CHART_HEIGHT = 350
    MONTHS = ['Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
    
    # Thresholds
    ABNORMAL_THRESHOLD = 20  # % change threshold for abnormal detection
    
    # Colors
    COLORS = {
        'primary': '#3b82f6',
        'success': '#16a34a',
        'danger': '#dc2626',
        'warning': '#f59e0b',
        'info': '#0ea5e9',
        'positive_bg': '#f0fdf4',
        'negative_bg': '#fef2f2',
    }
        
    profile_config = {
        "strict": {
            "daily": {
                "default": {"window": 90, "threshold": 3.0},
                "iqr": {"multiplier": 1.5},
                "z_score": {"threshold": 3.0},
                "lof": {"n_neighbors": 10, "contamination": 0.05},
                "change_point": {"pen": 5},
                "rolling_window_shift": {"window": 7, "alpha": 0.01, "detection_size": 5, "threshold": 0.6},
                "magnitude_test": {"window": 30, "threshold": 3.0},
                "contextual_consistency": {"window": 30, "threshold": 50},
                "persistence_test": {'persistence_window':7},
                "seasonal_test": {"seasonal_period": 7, "threshold": 3.0},
                "robustness_test": {"noise_std": 0.05, "window": 30, "runs": 5, "threshold": 3.0},
                "maxima_minima": {"window": 30},
                "trend_reversal": {"lookback": 7, "mode": "strict", "min_change_pct": 1.0},
                "perc_calculation": {"calculation_lookback": [1,7,30,365]}
            },
            "weekly": {
                "default": {"window": 52, "threshold": 3.0},
                "iqr": {"multiplier": 1.5},
                "z_score": {"threshold": 3.0},
                "lof": {"n_neighbors": 10, "contamination": 0.05},
                "change_point": {"pen": 5},
                "rolling_window_shift": {"window": 15, "alpha": 0.01, "detection_size": 5, "threshold": 0.6},
                "magnitude_test": {"window": 26, "threshold": 3.0},
                "contextual_consistency": {"window": 26, "threshold": 50},
                "persistence_test": {'persistence_window':4},
                "seasonal_test": {"seasonal_period": 7, "threshold": 3.0},
                "robustness_test": {"noise_std": 0.05, "window": 26, "runs": 5, "threshold": 3.0},
                "maxima_minima": {"window": 26},
                "trend_reversal": {"lookback": 7, "mode": "strict", "min_change_pct": 1.0},
                "perc_calculation": {"calculation_lookback": [1,4,12,26,52]}
            },
            "monthly": {
                "default": {"window": 12, "threshold": 3.0},
                "iqr": {"multiplier": 1.5},
                "z_score": {"threshold": 3.0},
                "lof": {"n_neighbors": 10, "contamination": 0.05},
                "change_point": {"pen": 5},
                "rolling_window_shift": {"window": 6, "alpha": 0.01, "detection_size": 5, "threshold": 0.6},
                "magnitude_test": {"window": 26, "threshold": 3.0},
                "contextual_consistency": {"window": 26, "threshold": 50},
                "persistence_test": {'persistence_window':7},
                "seasonal_test": {"seasonal_period": 7, "threshold": 3.0},
                "robustness_test": {"noise_std": 0.05, "window": 26, "runs": 5, "threshold": 3.0},
                "maxima_minima": {"window": 26},
                "trend_reversal": {"lookback": 7, "mode": "strict", "min_change_pct": 1.0},
                "perc_calculation": {"calculation_lookback": [1,4,12,26,52]}
            }
        },

        "relaxed": {
            "daily": {
                "default": {"window": 30, "threshold": 3.0},
                "iqr": {"multiplier": 1.5},
                "z_score": {"threshold": 3.0},
                "lof": {"n_neighbors": 10, "contamination": 0.05},
                "change_point": {"pen": 5},
                "rolling_window_shift": {"window": 7, "alpha": 0.01, "detection_size": 5, "threshold": 0.6},
                "magnitude_test": {"window": 30, "threshold": 2.5},
                "contextual_consistency": {"window": 30, "threshold": 50},
                "persistence_test": {'persistence_window':7},
                "seasonal_test": {"seasonal_period": 7, "threshold": 3.0},
                "robustness_test": {"noise_std": 0.05, "window": 30, "runs": 5, "threshold": 3.0},
                "maxima_minima": {"window": 30},
                "trend_reversal": {"lookback": 7, "mode": "strict", "min_change_pct": 1.0},
                "perc_calculation": {"calculation_lookback": [1,7,30,365]}
            },
            "weekly": {
                "default": {"window": 52, "threshold": 3.0},
                "iqr": {"multiplier": 1.5},
                "z_score": {"threshold": 3.0},
                "lof": {"n_neighbors": 10, "contamination": 0.05},
                "change_point": {"pen": 5},
                "rolling_window_shift": {"window": 15, "alpha": 0.01, "detection_size": 5, "threshold": 0.6},
                "magnitude_test": {"window": 26, "threshold": 3.0},
                "contextual_consistency": {"window": 26, "threshold": 50},
                "persistence_test": {'persistence_window':7},
                "seasonal_test": {"seasonal_period": 7, "threshold": 3.0},
                "robustness_test": {"noise_std": 0.05, "window": 26, "runs": 5, "threshold": 3.0},
                "maxima_minima": {"window": 26},
                "trend_reversal": {"lookback": 7, "mode": "strict", "min_change_pct": 1.0},
                "perc_calculation": {"calculation_lookback": [1,4,12,26,52]}
            },
            "monthly": {
                "default": {"window": 52, "threshold": 3.0},
                "iqr": {"multiplier": 1.5},
                "z_score": {"threshold": 3.0},
                "lof": {"n_neighbors": 10, "contamination": 0.05},
                "change_point": {"pen": 5},
                "rolling_window_shift": {"window": 15, "alpha": 0.01, "detection_size": 5, "threshold": 0.6},
                "magnitude_test": {"window": 26, "threshold": 3.0},
                "contextual_consistency": {"window": 26, "threshold": 50},
                "persistence_test": {'persistence_window':7},
                "seasonal_test": {"seasonal_period": 7, "threshold": 3.0},
                "robustness_test": {"noise_std": 0.05, "window": 26, "runs": 5, "threshold": 3.0},
                "maxima_minima": {"window": 26},
                "trend_reversal": {"lookback": 7, "mode": "strict", "min_change_pct": 1.0},
                "perc_calculation": {"calculation_lookback": [1,4,12,26,52]}
            }
        },

        "aggressive": {
            "daily": {
                "default": {"window": 90, "threshold": 3.0},
                "iqr": {"multiplier": 1.5},
                "z_score": {"threshold": 3.0},
                "lof": {"n_neighbors": 10, "contamination": 0.05},
                "change_point": {"pen": 5},
                "rolling_window_shift": {"window": 7, "alpha": 0.01, "detection_size": 5, "threshold": 0.6},
                "magnitude_test": {"window": 30, "threshold": 3.0},
                "contextual_consistency": {"window": 30, "threshold": 50},
                "persistence_test": {'persistence_window':7},
                "seasonal_test": {"seasonal_period": 7, "threshold": 3.0},
                "robustness_test": {"noise_std": 0.05, "window": 30, "runs": 5, "threshold": 3.0},
                "maxima_minima": {"window": 30},
                "trend_reversal": {"lookback": 7, "mode": "strict", "min_change_pct": 1.0},
                "perc_calculation": {"calculation_lookback": [1,7,30,365]}
            },
            "weekly": {
                "default": {"window": 52, "threshold": 3.0},
                "iqr": {"multiplier": 1.5},
                "z_score": {"threshold": 3.0},
                "lof": {"n_neighbors": 10, "contamination": 0.05},
                "change_point": {"pen": 5},
                "rolling_window_shift": {"window": 15, "alpha": 0.01, "detection_size": 5, "threshold": 0.6},
                "magnitude_test": {"window": 26, "threshold": 3.0},
                "contextual_consistency": {"window": 26, "threshold": 50},
                "persistence_test": {'persistence_window':7},
                "seasonal_test": {"seasonal_period": 7, "threshold": 3.0},
                "robustness_test": {"noise_std": 0.05, "window": 26, "runs": 5, "threshold": 3.0},
                "maxima_minima": {"window": 26},
                "trend_reversal": {"lookback": 7, "mode": "strict", "min_change_pct": 1.0},
                "perc_calculation": {"calculation_lookback": [1,4,12,26,52]}
            },
            "monthly": {
                "default": {"window": 52, "threshold": 3.0},
                "iqr": {"multiplier": 1.5},
                "z_score": {"threshold": 3.0},
                "lof": {"n_neighbors": 10, "contamination": 0.05},
                "change_point": {"pen": 5},
                "rolling_window_shift": {"window": 15, "alpha": 0.01, "detection_size": 5, "threshold": 0.6},
                "magnitude_test": {"window": 26, "threshold": 3.0},
                "contextual_consistency": {"window": 26, "threshold": 50},
                "persistence_test": {'persistence_window':7},
                "seasonal_test": {"seasonal_period": 7, "threshold": 3.0},
                "robustness_test": {"noise_std": 0.05, "window": 26, "runs": 5, "threshold": 3.0},
                "maxima_minima": {"window": 26},
                "trend_reversal": {"lookback": 7, "mode": "strict", "min_change_pct": 1.0},
                "perc_calculation": {"calculation_lookback": [1,4,12,26,52]}
            }
        }
    }
    

    @staticmethod
    def get_params(profile='strict', frequency='daily', method = None):
        """
        Extracts config parameters for a specific
        profile + frequency + method.

        Example:
          get_params("strict", "daily", "trend_reversal")
        """
        try:
            block = profile_config[profile][frequency]
            return block if method is None else block[method]
        except KeyError as e:
            raise ValueError(f"Invalid config lookup: {profile}/{frequency}/{method}") from e

