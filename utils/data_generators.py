import random
import pandas as pd


# Function to generate mock time series data for 12 months
def generate_mock_data(kpi_name, seed=42):
    if seed:
        random.seed(hash(kpi_name + str(seed)))
    
    months = ['Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
    base_value = random.randint(3000, 5000)
    values = []
    
    for i in range(12):
        variation = random.randint(-500, 800)
        values.append(base_value + variation)
    
    change_pct = random.uniform(-25, 25)
    
    return {
        'months': months,
        'values': values,
        'current_value': values[-1],
        'change_pct': change_pct,
        'vs_last': f"vs last 12m"
    }

# Function to generate breakdown data
def generate_breakdown_data(dimension, kpi_name):
    if dimension == "Country":
        items = ["USA", "India", "Germany", "Japan", "UK", "France", "Canada", "Australia"]
    elif dimension == "Channel":
        items = ["E-commerce", "Retail", "Wholesale", "Direct", "Partner", "Marketplace", "Social Media"]
    elif dimension == "Region":
        items = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East", "Africa"]
    else:
        items = ["Category A", "Category B", "Category C", "Category D", "Category E"]
    
    data = []
    for item in items:
        contrib = random.randint(5, 40)
        data.append({
            "FACTOR": item,
            "VALUE": f"{random.uniform(10, 50):.1f}M",
            "% CHANGE": random.randint(-10, 20),
            "% CONTRIB": f"{contrib}%"
        })
    
    return pd.DataFrame(data)

# Function to generate causal drivers
def generate_causal_drivers(kpi_name):
    drivers = [
        {
            "title": "Holiday Season",
            "impact": random.randint(85, 98),
            "description": f"Seasonal increase in demand during Q4 contributed significantly to {kpi_name.lower()} growth."
        },
        {
            "title": "Marketing Campaign",
            "impact": random.randint(70, 85),
            "description": f"New digital marketing campaign launched in October drove customer acquisition by 15%."
        },
        {
            "title": "Product Launch",
            "impact": random.randint(65, 80),
            "description": f"New product features released in August improved user engagement metrics."
        }
    ]
    
    return drivers
