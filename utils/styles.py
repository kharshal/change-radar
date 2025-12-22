# ============================================================================
# FILE: utils/styles.py
# ============================================================================
"""
CSS styles for the application.
"""

def get_custom_css() -> str:
    """Return custom CSS styles."""
    return """
    <style>
        .main-header {
            font-size: 24px;
            font-weight: 600;
            color: #1f2937;
            padding: 20px 0;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 30px;
        }
        
        .page-header {
            padding: 0;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .page-header h1 {
            font-size: 24px;
            font-weight: 600;
            color: #1f2937;
            margin: 0;
        }
        
        .page-header h3 {
            font-size: 16px;
            font-weight: 300;
            color: #374151;
            margin: -12px 0 0 0;
        }
        
        .upload-box {
            border-radius: 8px;
            padding: 0 0 10px 0;
            text-align: center;
            background: #f3f4f6;
            margin: 0;
        }
        
        .success-box {
            background: #f0fdf4;
            border: 1px solid #86efac;
            border-radius: 8px;
            padding: 24px;
            text-align: center;
            margin: 20px 0;
        }
        
        .success-text {
            color: #16a34a;
            font-weight: 600;
            font-size: 16px;
        }
        
        .kpi-card {
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .kpi-card-positive {
            background: #f0fdf4;
            border-color: #86efac;
        }
        
        .kpi-card-negative {
            background: #fef2f2;
            border-color: #fecaca;
        }
        
        .kpi-card-selected {
            border: 3px solid #3b82f6;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
        }
        
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        .narrative-box {
            background: #fef9c3;
            border: 1px solid #fde047;
            border-radius: 8px;
            padding: 5px 20px;
            margin: 0 0 20px 0;
            font-size: 15px;
            line-height: 1.6;
        }
        
        .driver-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        
        div[data-testid="stFileUploader"] {
            margin-top: -40px;
            padding-bottom: 20px;
        }
        
        div[data-testid="stFileUploader"] section {
            display: flex;
            justify-content: center;
        }
    </style>
    """
