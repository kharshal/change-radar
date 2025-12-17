import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

# Page configuration
st.set_page_config(
    page_title="Change Radar",
    page_icon="🧠",
    layout="wide"
)

# st.columns([1, 6])[0].text("🧠 Change Radar")

st.columns([1, 6])[0].markdown(
    """
    <div style="font-size:16px; font-weight:600; color:#111827;">
        🧠 Change Radar
    </div>
    """,
    unsafe_allow_html=True
)

# Custom CSS for styling
st.markdown("""
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
        border: 2px dashed #93c5fd;
        border-radius: 8px;
        padding: 0;
        text-align: center;
        background: #eff6ff;
        margin: 0;
    }
    
    .upload-box-purple {
        border: 2px dashed #c4b5fd;
        background: #f5f3ff;
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
        display: flex;
        justify-content: space-between;
        align-items: center;
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
    
    .kpi-name {
        font-size: 14px;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 8px;
    }
    
    .kpi-value {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 4px;
    }
    
    .kpi-change-positive {
        color: #16a34a;
        font-size: 14px;
        font-weight: 600;
    }
    
    .kpi-change-negative {
        color: #dc2626;
        font-size: 14px;
        font-weight: 600;
    }
        
    .narrative-box {
        background: #fef9c3;
        border: 1px solid #fde047;
        border-radius: 8px;
        padding: 5px 20px 5px 20px;
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
    
    .driver-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .impact-badge {
        background: #f3e8ff;
        color: #7c3aed;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .driver-desc {
        color: #6b7280;
        font-size: 14px;
        line-height: 1.5;
    }
    
    .show-more {
        color: #7c3aed;
        font-weight: 600;
        cursor: pointer;
        font-size: 14px;
        margin-top: 8px;
    }
    
    div[data-testid="stDataFrame"] {
        height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dkl_generated' not in st.session_state:
    st.session_state.dkl_generated = False
if 'fkl_uploaded' not in st.session_state:
    st.session_state.fkl_uploaded = False
if 'causal_graph_generated' not in st.session_state:
    st.session_state.causal_graph_generated = False
if 'kpi_list' not in st.session_state:
    st.session_state.kpi_list = []
if 'selected_kpis' not in st.session_state:
    st.session_state.selected_kpis = []
if 'active_kpi' not in st.session_state:
    st.session_state.active_kpi = None
if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False
if 'show_deep_dive' not in st.session_state:
    st.session_state.show_deep_dive = False
if 'deep_dive_driver' not in st.session_state:
    st.session_state.deep_dive_driver = None
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False


# Function to extract KPI names from Excel files
def extract_kpis_from_excel(files):
    kpi_names = []
    for file in files:
        try:
            if file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
                if 'KPI Name' in df.columns:
                    kpi_names.extend(df['KPI Name'].dropna().unique().tolist())
                elif 'kpi name' in df.columns:
                    kpi_names.extend(df['kpi name'].dropna().unique().tolist())
                elif 'KPI_Name' in df.columns:
                    kpi_names.extend(df['KPI_Name'].dropna().unique().tolist())
        except Exception as e:
            st.warning(f"Could not extract KPIs from {file.name}: {str(e)}")
    
    return list(set(kpi_names))

# Function to generate mock time series data for 12 months
def generate_mock_data(kpi_name, seed=None):
    if seed:
        random.seed(hash(kpi_name + str(seed)))
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
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
            "description": f"New digital marketing campaign launched in November drove customer acquisition by 15%."
        },
        {
            "title": "Product Launch",
            "impact": random.randint(65, 80),
            "description": f"New product features released in October improved user engagement metrics."
        }
    ]
    
    return drivers

# Deep Dive Page
if st.session_state.show_deep_dive:
    col1, col2, _ = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back"):
            st.session_state.show_deep_dive = False
            st.rerun()
    
    with col2:
        st.title(f"KPI Deep Dive: {st.session_state.deep_dive_driver['title']}")    
    
    st.markdown("---")
        
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Relationship Graph")
        
        # Mock causal graph using Plotly
        fig = go.Figure()
        
        # Add nodes
        nodes = ["Marketing Spend", "Customer Traffic", st.session_state.active_kpi, 
                 st.session_state.deep_dive_driver['title'], "Conversion Rate"]
        x_pos = [0.2, 0.5, 0.8, 0.3, 0.7]
        y_pos = [0.8, 0.5, 0.5, 0.2, 0.2]
        
        # Add edges
        edges = [(0, 1), (0, 3), (3, 4), (4, 1), (1, 2)]
        
        for edge in edges:
            fig.add_trace(go.Scatter(
                x=[x_pos[edge[0]], x_pos[edge[1]]],
                y=[y_pos[edge[0]], y_pos[edge[1]]],
                mode='lines',
                line=dict(color='#3b82f6', width=3),
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
                arrowcolor='#3b82f6'
            )
        
        # Add nodes
        for i, node in enumerate(nodes):
            if node == st.session_state.deep_dive_driver['title']:
                color = '#7c3aed'
            elif node == st.session_state.active_kpi:
                color = '#ef4444'
            else:
                color = '#3b82f6'
            
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
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
#         st.subheader("Dependencies Explanation")
        
        st.markdown(f"""
        #### Impact Analysis: {st.session_state.deep_dive_driver['title']}
        
        **Overall Impact**: {st.session_state.deep_dive_driver['impact']}% contribution to {st.session_state.active_kpi}
        
        {st.session_state.deep_dive_driver['description']}
        
        #### Detailed Causal Chain
        
        **1. Primary Driver**
        - {st.session_state.deep_dive_driver['title']} showed significant variation in the analyzed period
        - Direct correlation coefficient with {st.session_state.active_kpi}: 0.{random.randint(75, 95)}
        - Time lag effect: {random.randint(3, 14)} days
        
        **2. Intermediate Effects**
        - Customer traffic increased by {random.randint(10, 25)}%
        - Conversion rate improved by {random.randint(5, 15)}%
        - Average transaction value changed by {random.randint(-5, 20)}%
        
        **3. Cascading Impact**
        - The primary driver influenced multiple downstream metrics
        - Secondary effects contributed an additional {random.randint(15, 30)}% impact
        - Cross-functional dependencies amplified the overall effect
        
        #### Statistical Validation
        
        - **Granger Causality Test**: p-value < 0.01 (statistically significant)
        - **Cross-Correlation**: Peak at lag {random.randint(1, 7)} days
        - **Variance Explained**: {random.randint(40, 70)}% of total variation
        
        #### Recommendations
        
        1. Monitor {st.session_state.deep_dive_driver['title']} closely as a leading indicator
        2. Implement early warning systems for significant deviations
        3. Optimize response strategies to leverage positive trends
        4. Develop contingency plans for negative scenarios
        5. Continue tracking correlation strength over time
        
        #### Data Quality Notes
        
        - Analysis based on {random.randint(90, 365)} days of historical data
        - Confidence interval: {random.randint(90, 98)}%
        - Missing data points: < {random.randint(1, 5)}%
        """)
    
    st.stop()

# Insight Dashboard Page
if st.session_state.show_dashboard:
        
    col1, col2, _ = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back"):
            st.session_state.show_dashboard = False
            st.session_state.active_kpi = None
            st.rerun()
    
    with col2:
        st.title("Insights Dashboard")
    
    st.markdown("---")
    
    # Set first KPI as active if not set
    if st.session_state.active_kpi is None and st.session_state.selected_kpis:
        st.session_state.active_kpi = st.session_state.selected_kpis[0]
    
    # Generate mock data for all selected KPIs
    kpi_data = {}
    for kpi in st.session_state.selected_kpis:
        kpi_data[kpi] = generate_mock_data(kpi, seed=42)
    
    # Top Section: KPIs and Graph
    col_left, col_right = st.columns([1, 2])
    
    with col_left:

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

            # Scrollable KPI list
            for i, kpi in enumerate(st.session_state.selected_kpis):
                data = kpi_data[kpi]
                change_pct = data['change_pct']
                is_positive = change_pct > 0
                is_active = kpi == st.session_state.active_kpi

                # Determine card styling
                card_bg_class = "kpi-card-positive" if is_positive else "kpi-card-negative"
                card_class = f"kpi-card {card_bg_class}"
                if is_active:
                    card_class += " kpi-card-selected"

                change_class = "kpi-change-positive" if is_positive else "kpi-change-negative"
                arrow_symbol = "↑" if is_positive else "↓"
                change_symbol = "+" if is_positive else ""

                # Create unique key for each KPI button
                button_key = f"select_kpi_{i}_{kpi}"

                dot = "🟢" if is_positive else "🔴"
                arrow = "▲" if is_positive else "▼"
                sign = "+" if is_positive else ""

                button_label = f"{kpi}  | {dot} {sign}{abs(change_pct):.1f}%  {arrow}"
                
                # Just make the button look like the card
                if st.button(
                    button_label,  # Show the KPI name
                    key=button_key,
                    use_container_width=True,
                    help=f"Click to analyze {kpi}"
                ):
                    st.session_state.active_kpi = kpi
                    st.rerun()
                    
#     # Working Code with Radio Button
#     with col_left:
#         st.markdown("### 📊 Analyzed KPIs")

#         # Default active KPI
#         if "active_kpi" not in st.session_state and st.session_state.selected_kpis:
#             st.session_state.active_kpi = st.session_state.selected_kpis[0]

#         # Build radio labels
#         kpi_labels = []
#         label_to_kpi = {}

#         for kpi in st.session_state.selected_kpis:
#             data = kpi_data[kpi]
#             arrow = "↑" if data["change_pct"] > 0 else "↓"
#             sign = "+" if data["change_pct"] > 0 else ""
#             label = f"{kpi}   {arrow} {sign}{abs(data['change_pct']):.1f}%"
#             kpi_labels.append(label)
#             label_to_kpi[label] = kpi

#         selected_label = st.radio(
#             label="",
#             options=kpi_labels,
#             index=kpi_labels.index(
#                 next(l for l in kpi_labels if label_to_kpi[l] == st.session_state.active_kpi)
#             ),
#             label_visibility="collapsed"
#         )

#         st.markdown('</div>', unsafe_allow_html=True)

#         # Update active KPI
#         st.session_state.active_kpi = label_to_kpi[selected_label]


    
    with col_right:
        with st.container(height=500):

            if st.session_state.active_kpi:
                active_data = kpi_data[st.session_state.active_kpi]

                st.markdown(
                    f"""
                    <div style="border:1px solid #ccc; padding:12px 16px; 
                        border-radius:6px; background-color:#F3F4F6;
                        color:#111827; font-family:system-ui, sans-serif; margin: -8px 0 8px 0 ;">
                        <div style="font-size:16px; font-weight:600;">📈 {st.session_state.active_kpi} Chart</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    #             st.markdown(f"### 📈 {st.session_state.active_kpi} Trend (12 Months)")

                # Display current value and change
                change_color = "#16a34a" if active_data['change_pct'] > 0 else "#dc2626"

    #             col_val1, col_val2 = st.columns([1, 1])

    #             with col_val1:
    #                 st.markdown(f"<div style='text-align: left; font-size: 48px; font-weight: 700;'>{active_data['current_value']:,}</div>", unsafe_allow_html=True)
    #             with col_val2:
    #                 st.markdown(f"<div style='text-align: right; color: {change_color}; font-size: 24px; font-weight: 600; margin-top: 12px;'>{'+' if active_data['change_pct'] > 0 else ''}{active_data['change_pct']:.1f}% (12m)</div>", unsafe_allow_html=True)

                # Create line chart
                fig = go.Figure()

                colors = ['#3b82f6'] * 11 + ['#ef4444']  # Last point in red

                fig.add_trace(go.Scatter(
                    x=active_data['months'],
                    y=active_data['values'],
                    mode='lines+markers',
                    line=dict(color='#3b82f6', width=3),
                    marker=dict(size=8, color=colors, line=dict(color='white', width=2)),
                    hovertemplate='<b>%{x}</b><br>Value: %{y:,.0f}<extra></extra>'
                ))

                # Add average line
                avg_value = sum(active_data['values']) / len(active_data['values'])
                fig.add_hline(
                    y=avg_value, 
                    line_dash="dot", 
                    line_color="#22c55e", 
                    annotation_text=f"Avg: {avg_value:,.0f}", 
                    annotation_position="right"
                )

                fig.update_layout(
                    height=350,
                    xaxis_title="",
                    yaxis_title="",
                    plot_bgcolor='white',
                    hovermode='x unified',
                    margin=dict(l=20, r=20, t=20, b=40),
                    xaxis=dict(showgrid=True, gridcolor='#f3f4f6'),
                    yaxis=dict(showgrid=True, gridcolor='#f3f4f6')
                )

                st.plotly_chart(fig, use_container_width=True)
                
                # Feedback button
                col_fb1, col_fb2, col_fb3 = st.columns([3, 1, 1])
                with col_fb3:
                    st.button("💬 Give Feedback", use_container_width=True)
#                         st.session_state.show_feedback = True

    
#     with st.container(height=150):

    if st.session_state.active_kpi:
        active_data = kpi_data[st.session_state.active_kpi]

        show_alert = abs(active_data["change_pct"]) > 20

        direction = "above" if active_data['change_pct'] > 0 else "below"
        abs_change = abs(active_data['change_pct'])
        shift_detected = "has detected an abnormal shift" if abs(active_data['change_pct']) > 20 else "shows normal variation"

        narrative = f"""{st.session_state.active_kpi} for Dec 2025, is {abs_change:.1f}% {direction} the yearly average between Jan 2025 and Nov 2025. Value of {active_data['current_value']} reported for Dec 2025, is 220% higher than what was reported during Dec 2024, and 70% higher than the reported {st.session_state.active_kpi} in Nov 2025."""

    # Generate alert badge if needed
    alert_badge = ""
    if show_alert:
        alert_badge = '''
        <span style="
            display: inline-block;
            background: #fee2e2;
            color: #dc2626;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 10px;
            vertical-align: middle;
        ">
            🔴 Abnormal    
        </span>
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

#     st.markdown("---")

    
    # Bottom Section: Impact Factors and Causal Drivers
    col_left_bottom, col_right_bottom = st.columns([1, 1])
    
    with col_left_bottom:
        
#         st.info(f"### 🧩 Where all {st.session_state.active_kpi} has changed")
        
        st.markdown(
            f"""
            <div style="
                font-size:24px; font-weight:600;
                color:#1f2937; margin-bottom:0px;
                padding:12px 16px;
                border-radius:6px; background-color:#BFDBFE;
            ">🧩 Where all {st.session_state.active_kpi} has changed
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

            if st.session_state.active_kpi:
                # Generate and display breakdown table
                df_breakdown = generate_breakdown_data(dimension, st.session_state.active_kpi)

                # Custom styling for the dataframe
                def color_change(val):
                    if isinstance(val, (int, float)):
                        color = '#16a34a' if val > 0 else '#dc2626'
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
#                     height=350
                )

                st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right_bottom:
#         st.info(f"### 🔍 Why {st.session_state.active_kpi} has changed")
        
        st.markdown(
            f"""
            <div style="
                font-size:24px; font-weight:600;
                color:#1f2937; margin-bottom:0px;
                padding:12px 16px;
                border-radius:6px; background-color:#BFDBFE;
            ">🔍 Why {st.session_state.active_kpi} has changed
            </div>
            """,
            unsafe_allow_html=True
        )
                
        with st.container(height=500):

            if st.session_state.active_kpi:
                drivers = generate_causal_drivers(st.session_state.active_kpi)

#                 col_d1, col_d2 = st.columns([4,1])

#                 for i, driver in enumerate(drivers):

#                     with col_d1:
#                         st.markdown(
#                             f"""
#                             <div class="driver-card">
#                                 <div class="driver-desc">{driver['description']}</div>
#                             </div>
#                             """,
#                             unsafe_allow_html=True
#                         )

#                     with col_d2:            
#                         if st.button("Show More →", key=f"driver_{i}", use_container_width=True):
#                             st.session_state.show_deep_dive = True
#                             st.session_state.deep_dive_driver = driver
#                             st.rerun()
                
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
                    # Add spacing to align button with card
                    st.markdown('<div style="margin:-70px 0px -20px -10px; min-height: 80px; display: flex; align-items: center;">', unsafe_allow_html=True)
                    if st.button("Show More →", key=f"driver_{i}", use_container_width=True):
                        st.session_state.show_deep_dive = True
                        st.session_state.deep_dive_driver = driver
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)                
                
    
    st.stop()

    
    
left, center, right = st.columns([1, 2, 1])

with center:
    
    # Main Dashboard Header
    st.markdown(
        """
        <div class="page-header">
            <h1>Build Knowledge Layer</h1>
            <h3>Upload data and process documents to create insight dashboard</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Section 1: Data Knowledge Generator
    with st.expander("1️⃣ Data Knowledge Layer", expanded=True):
        if not st.session_state.dkl_generated:
            st.markdown("""
            <div class="upload-box">
                <h3 style="color: #2563eb; margin: 8px 0 8px 0;">📊 Upload your Data</h3>
                <!-- <p style="color: #6b7280;">CSV, Excel, or JSON files</p> -->
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <style>
            /* Move uploader visually into upload-box */
            div[data-testid="stFileUploader"] {
                margin-top: -40px;
                padding-bottom: 20px;
            }

            /* Center uploader button */
            div[data-testid="stFileUploader"] section {
                display: flex;
                justify-content: center;
            }
            </style>
            """, unsafe_allow_html=True)
            
            dkl_files = st.file_uploader(
                "",
                accept_multiple_files=True,
                type=['csv', 'xlsx', 'json'],
                key="dkl_uploader"
            )

            st.markdown("**Provide Data Description**")
            business_context = st.text_area(
                "Description",
                placeholder="Describe about data to enhance generation accuracy...",
                height=80,
                label_visibility="collapsed"
            )

            if st.button("Submit", type="primary", key="dkl_submit", use_container_width=True):
                if dkl_files:
                    with st.spinner("Generating Data Knowledge Layer..."):
                        time.sleep(2)
                    st.session_state.dkl_generated = True
                    st.rerun()
                else:
                    st.error("Please upload data files first!")
        else:
            st.markdown("""
            <div class="success-box">
                <p class="success-text">✅ Data Knowledge Layer Successfully Generated</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download",
                    data="Mock DKL Data",
                    file_name="dkl_output.json",
                    use_container_width=True
                )
            with col2:
                if st.button("⬆️ Upload", use_container_width=True): # 🔄 Generate Again
                    st.session_state.dkl_generated = False
                    # st.rerun()

    # Section 2: Upload FKL
    with st.expander("2️⃣ Business Knowledge Layer", expanded=True):
        if not st.session_state.fkl_uploaded:
            st.markdown("""
            <div class="upload-box upload-box-purple">
                <h4 style="color: #7c3aed; margin: 14px 20px 8px 20px;">📄 Upload business process and functional documents</h4>
                <!-- <p style="color: #6b7280;">DOCX, PDF, Text, CSV, Excel, JSON</p> -->
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <style>
            /* Move uploader visually into upload-box */
            div[data-testid="stFileUploader"] {
                margin-top: -40px;
                padding-bottom: 20px;
            }

            /* Center uploader button */
            div[data-testid="stFileUploader"] section {
                display: flex;
                justify-content: center;
            }
            </style>
            """, unsafe_allow_html=True)
            
            
            fkl_files = st.file_uploader(
                "",
                accept_multiple_files=True,
                type=['docx', 'pdf', 'txt', 'csv', 'xlsx', 'json'],
                key="fkl_uploader",
                disabled=not st.session_state.dkl_generated
            )

            if st.button("Submit", type="primary", key="fkl_submit",
                         use_container_width=True, disabled=not st.session_state.dkl_generated):
                if fkl_files:
                    with st.spinner("Processing business process files and extracting KPIs..."):
                        time.sleep(1.5)
                        kpi_names = extract_kpis_from_excel(fkl_files)

                        if kpi_names:
                            st.session_state.kpi_list = kpi_names
                            st.session_state.fkl_uploaded = True
                            st.success(f"✅ Found {len(kpi_names)} KPIs from business process files!")
                            st.rerun()
                        else:
                            st.error("No 'KPI Name' column found in Excel files. Please ensure your Excel file has a column named 'KPI Name'.")
                else:
                    st.error("Please upload business process files first!")
        else:
            
            st.markdown("""
            <div class="success-box">
                <p class="success-text">✅ Uploaded business documents are successfully parsed and accepted!</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"Extracted {len(st.session_state.kpi_list)} KPIs from uploaded files")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download",
                    data="Mock FKL Data",
                    file_name="fkl_output.json",
                    use_container_width=True
                )
            with col2:
                if st.button("⬆️ Upload", key="reupload_fkl", use_container_width=True):
                    st.session_state.fkl_uploaded = False
                    st.session_state.kpi_list
                    st.rerun()

                    
    # Section 3: Causal Graph Generation
    with st.expander("3️⃣ KPI Relationship Layer", expanded=True):
        if not st.session_state.causal_graph_generated:
            st.info("Extract KPI dependencies and relationships from the data")
            if st.button("Submit", type="primary", use_container_width=True, disabled=not st.session_state.fkl_uploaded):
                with st.spinner("This may take a few moments...System is extracting dependencies for you"):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.05)
                        progress_bar.progress(i + 1)
                st.session_state.causal_graph_generated = True
                st.rerun()
        else:
            st.markdown("""
            <div class="success-box">
                <p class="success-text">✅ Causal Graph Generated Successfully</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download",
                    data="Mock Causal Graph Data",
                    file_name="causal_graph.json",
                    use_container_width=True
                )
                
            with col2:
                st.button("⬆️ Upload", key="reupload_graph", use_container_width=True)
    #             uploaded_graph = st.file_uploader("📤 Upload Graph", type=['json', 'xml'], label_visibility="collapsed", key="graph_upload")
#             with col3:
#                 if st.button("🔄 Regenerate", use_container_width=True):
#                     st.session_state.causal_graph_generated = False
#                     st.rerun()


    # Section 4: KPI Selection & Analysis
    with st.expander("4️⃣ KPI Selection", expanded=True):
        if st.session_state.causal_graph_generated and st.session_state.kpi_list:
            # Check if all KPIs are currently selected
            all_selected = len(st.session_state.selected_kpis) == len(st.session_state.kpi_list)
            some_selected = len(st.session_state.selected_kpis) > 0

            # Master checkbox for Select All
            col_master, col_label = st.columns([2, 1])
            with col_label:
                # Master checkbox
                master_checked = st.checkbox(
                    "Select All",
                    value=all_selected,
                    key="master_select_all",
                    help="Select/Deselect all KPIs"
                )
                if master_checked != all_selected:
                    if master_checked:
                        st.session_state.selected_kpis = st.session_state.kpi_list.copy()
                    else:
                        st.session_state.selected_kpis = []
                    st.rerun()

            with col_master:
                st.markdown(f"**Select KPIs you want to monitor**", unsafe_allow_html=True)

            # Individual KPI checkboxes
            for kpi in st.session_state.kpi_list:
                is_selected = kpi in st.session_state.selected_kpis

                if st.checkbox(kpi, value=is_selected, key=f"kpi_{kpi}"):
                    if kpi not in st.session_state.selected_kpis:
                        st.session_state.selected_kpis.append(kpi)
                else:
                    if kpi in st.session_state.selected_kpis:
                        st.session_state.selected_kpis.remove(kpi)

            if st.session_state.selected_kpis:
                st.info(f"✓ Selected {len(st.session_state.selected_kpis)} of {len(st.session_state.kpi_list)} KPI(s)")
                if st.button("🚀 Launch Insight Dashboard", type="primary", use_container_width=True):
                    st.session_state.show_dashboard = True
                    st.rerun()
            else:
                st.warning("⚠️ Please select at least one KPI to launch Insight Dashboard")
        else:
            st.info("Complete previous steps to enable KPI selection")