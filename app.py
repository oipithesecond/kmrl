# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
from solver import load_data, preprocess_data, preprocess_shunting_costs, create_and_solve_model

# --- Page Configuration ---
st.set_page_config(
    page_title="KMRL AI Train Scheduler",
    page_icon="🚊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern UI with Dark Mode Support ---
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header redesign with subtle gradient */
    .main-header {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .header-title {
        color: #6366f1;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        color: #64748b;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .main-header {
            background: linear-gradient(90deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
            border: 1px solid rgba(99, 102, 241, 0.3);
        }
        .header-title {
            color: #a5b4fc;
        }
        .header-subtitle {
            color: #94a3b8;
        }
    }
    
    /* Center tabs and make them full width */
    .stTabs {
        width: 100%;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 2rem;
        background: transparent;
        border-bottom: 2px solid rgba(99, 102, 241, 0.1);
        padding-bottom: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        max-width: 200px;
        height: 48px;
        background-color: transparent;
        border: none;
        border-radius: 12px 12px 0 0;
        color: #64748b;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(99, 102, 241, 0.05);
        color: #6366f1;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }
    
    /* Metric cards with better colors */
    .metric-container {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
        border-color: #cbd5e1;
    }
    
    /* Dark mode metric cards */
    @media (prefers-color-scheme: dark) {
        .metric-container {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        .metric-container:hover {
            border-color: rgba(99, 102, 241, 0.4);
        }
    }
    
    /* Status cards with softer colors */
    .status-card {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .status-card:hover {
        transform: translateX(4px);
    }
    
    .service-card {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 4px solid #22c55e;
        color: #15803d;
    }
    
    .standby-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        color: #92400e;
    }
    
    .maintenance-card {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
        color: #991b1b;
    }
    
    /* Dark mode status cards */
    @media (prefers-color-scheme: dark) {
        .service-card {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.25) 100%);
            color: #86efac;
        }
        .standby-card {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(245, 158, 11, 0.25) 100%);
            color: #fbbf24;
        }
        .maintenance-card {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.25) 100%);
            color: #fca5a5;
        }
    }
    
    /* Improved metric styling */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    @media (prefers-color-scheme: dark) {
        [data-testid="metric-container"] {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
    }
    
    /* Landing page cards */
    .feature-card {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.1);
        border-color: #6366f1;
    }
    
    @media (prefers-color-scheme: dark) {
        .feature-card {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        .feature-card:hover {
            border-color: rgba(99, 102, 241, 0.6);
        }
    }
    
    /* Remove default Streamlit padding for tabs */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 2rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(99, 102, 241, 0.03) 0%, rgba(139, 92, 246, 0.03) 100%);
    }
    
    @media (prefers-color-scheme: dark) {
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'data_frames' not in st.session_state:
    st.session_state.data_frames = None
if 'scenario' not in st.session_state:
    st.session_state.scenario = None

# --- Header Section (Redesigned) ---
st.markdown("""
<div class="main-header">
    <h1 class="header-title">🚊 KMRL AI-Driven Train Induction Planner</h1>
    <p class="header-subtitle">Intelligent Fleet Management & Optimization System</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration Panel")
    
    # Scenario Selection
    scenarios = [d for d in os.listdir('.') if os.path.isdir(d) and d not in ['venv', '__pycache__', '.git', '.streamlit']]
    
    if scenarios:
        selected_scenario = st.selectbox(
            "📁 Select Scenario",
            scenarios,
            index=scenarios.index('bottleneck_case') if 'bottleneck_case' in scenarios else 0,
            help="Choose the operational scenario to optimize"
        )
        
        st.markdown("---")
        
        # Advanced Options (expandable)
        with st.expander("🔧 Advanced Settings", expanded=False):
            solver_timeout = st.slider("Solver Timeout (seconds)", 10, 120, 30)
            show_detailed_logs = st.checkbox("Show Detailed Reasoning", value=False)
            enable_what_if = st.checkbox("Enable What-If Analysis", value=False)
        
        st.markdown("---")
        
        # Action Buttons
        col1, col2 = st.columns(2)
        with col1:
            run_optimization = st.button("🚀 Optimize", type="primary", use_container_width=True)
        with col2:
            clear_results = st.button("🔄 Clear", use_container_width=True)
        
        if clear_results:
            st.session_state.optimization_results = None
            st.session_state.data_frames = None
            st.rerun()
    else:
        st.error("No scenario folders found! Please create a scenario folder with data files.")
        selected_scenario = None
        run_optimization = False
    
    st.markdown("---")
    
    # Info Section with better styling
    st.markdown("### 📊 System Status")
    st.success("✅ All systems operational")
    st.info(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- Main Content Area ---
if run_optimization and selected_scenario:
    with st.spinner("🔄 Loading data and running optimization..."):
        # Load and process data
        data_frames = load_data(selected_scenario)
        
        if data_frames:
            eligibility_dict = preprocess_data(data_frames)
            shunting_costs_dict = preprocess_shunting_costs(data_frames.get("layout_costs"))
            solution_df = create_and_solve_model(data_frames, eligibility_dict, shunting_costs_dict)
            
            if solution_df is not None:
                st.session_state.optimization_results = solution_df
                st.session_state.data_frames = data_frames
                st.session_state.scenario = selected_scenario
                st.success("✅ Optimization completed successfully!")
            else:
                st.error("❌ No feasible solution found. Please check constraints.")
        else:
            st.error(f"❌ Failed to load data for scenario: {selected_scenario}")

# Display Results
if st.session_state.optimization_results is not None:
    solution_df = st.session_state.optimization_results
    data_frames = st.session_state.data_frames
    
    # --- Key Metrics Dashboard ---
    st.markdown("## 📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_trains = len(solution_df)
    in_service = len(solution_df[solution_df['Assigned Status'] == 'Revenue Service'])
    standby = len(solution_df[solution_df['Assigned Status'] == 'Standby'])
    maintenance = len(solution_df[solution_df['Assigned Status'] == 'Maintenance'])
    
    with col1:
        st.metric(
            label="Total Fleet",
            value=total_trains,
            delta=f"{in_service/total_trains*100:.1f}% operational" if total_trains > 0 else "0.0% operational"
        )
    
    with col2:
        st.metric(
            label="🟢 Revenue Service",
            value=in_service,
            delta=f"{in_service/total_trains*100:.0f}%" if total_trains > 0 else "0%"
        )
    
    with col3:
        st.metric(
            label="🟡 Standby",
            value=standby,
            delta=f"{standby/total_trains*100:.0f}%" if total_trains > 0 else "0%"
        )
    
    with col4:
        st.metric(
            label="🔴 Maintenance",
            value=maintenance,
            delta=f"{maintenance/total_trains*100:.0f}%" if total_trains > 0 else "0%"
        )
    
    st.markdown("---")
    
    # --- Centered Tabbed Interface ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚊 Fleet Status", "📊 Analytics", "🔍 Detailed View", "⚠️ Alerts", "📋 Reports"])
    
    with tab1:
        st.markdown("### Fleet Assignment Overview")
        
        # Create three columns for different statuses
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🟢 Revenue Service")
            service_trains = solution_df[solution_df['Assigned Status'] == 'Revenue Service']
            if not service_trains.empty:
                for _, train in service_trains.iterrows():
                    st.markdown(f"""
                    <div class="status-card service-card">
                        <strong>{train['Trainset ID']}</strong><br>
                        <small>Ready for passenger operations</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No trains in revenue service")
        
        with col2:
            st.markdown("#### 🟡 Standby")
            standby_trains = solution_df[solution_df['Assigned Status'] == 'Standby']
            if not standby_trains.empty:
                for _, train in standby_trains.iterrows():
                    st.markdown(f"""
                    <div class="status-card standby-card">
                        <strong>{train['Trainset ID']}</strong><br>
                        <small>Reserve capacity available</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No trains on standby")
        
        with col3:
            st.markdown("#### 🔴 Maintenance")
            maintenance_trains = solution_df[solution_df['Assigned Status'] == 'Maintenance']
            if not maintenance_trains.empty:
                for _, train in maintenance_trains.iterrows():
                    st.markdown(f"""
                    <div class="status-card maintenance-card">
                        <strong>{train['Trainset ID']}</strong><br>
                        <small>Scheduled for maintenance</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No trains in maintenance")
    
    with tab2:
        st.markdown("### 📊 Operational Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Status Distribution Pie Chart
            status_counts = solution_df['Assigned Status'].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=.3,
                marker=dict(colors=['#22c55e', '#f59e0b', '#ef4444']),
                textfont=dict(size=14, color='white'),
                textposition='inside',
                textinfo='percent+label'
            )])
            fig_pie.update_layout(
                title="Fleet Status Distribution",
                showlegend=True,
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Mileage Analysis
            if 'trainsets' in data_frames:
                trainsets_df = data_frames['trainsets']
                merged_df = pd.merge(
                    solution_df[['Trainset ID', 'Assigned Status']],
                    trainsets_df[['trainset_id', 'cumulative_mileage_km']],
                    left_on='Trainset ID',
                    right_on='trainset_id'
                )
                
                fig_box = go.Figure()
                for status in merged_df['Assigned Status'].unique():
                    data = merged_df[merged_df['Assigned Status'] == status]['cumulative_mileage_km']
                    color = {'Revenue Service': '#22c55e', 'Standby': '#f59e0b', 'Maintenance': '#ef4444'}.get(status, '#94a3b8')
                    fig_box.add_trace(go.Box(
                        y=data,
                        name=status,
                        marker_color=color,
                        boxmean='sd'
                    ))
                
                fig_box.update_layout(
                    title="Mileage Distribution by Status",
                    yaxis_title="Cumulative Mileage (km)",
                    showlegend=True,
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_box, use_container_width=True)
        
        # Additional metrics in cards
        st.markdown("### 🎯 Optimization Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_mileage = data_frames['trainsets']['cumulative_mileage_km'].mean()
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="color: #6366f1; margin: 0;">Average Fleet Mileage</h4>
                <p style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">{avg_mileage:,.0f} km</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if 'resources' in data_frames:
                ibl_capacity = data_frames['resources'][data_frames['resources']['resource_id'] == 'IBL_Bays']['available_capacity'].iloc[0]
                utilization = (maintenance/ibl_capacity)*100 if ibl_capacity > 0 else 0
                st.markdown(f"""
                <div class="metric-container">
                    <h4 style="color: #6366f1; margin: 0;">IBL Bay Utilization</h4>
                    <p style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">{maintenance}/{ibl_capacity}</p>
                    <small style="color: #64748b;">{utilization:.0f}% utilized</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if 'slas' in data_frames:
                active_slas = len(data_frames['slas'][data_frames['slas']['current_exposure_hours'] < data_frames['slas']['target_exposure_hours']])
                st.markdown(f"""
                <div class="metric-container">
                    <h4 style="color: #6366f1; margin: 0;">Active Branding SLAs</h4>
                    <p style="font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;">{active_slas}</p>
                    <small style="color: #64748b;">Contracts active</small>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 🔍 Detailed Train Information")
        
        # Search and Filter
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_train = st.text_input("🔍 Search Train ID", placeholder="e.g., KMRL-T01")
        with col2:
            filter_status = st.selectbox("Filter by Status", ["All"] + list(solution_df['Assigned Status'].unique()))
        with col3:
            show_reasoning = st.checkbox("Show Reasoning", value=False)
        
        # Apply filters
        display_df = solution_df.copy()
        if search_train:
            display_df = display_df[display_df['Trainset ID'].str.contains(search_train, case=False)]
        if filter_status != "All":
            display_df = display_df[display_df['Assigned Status'] == filter_status]
        
        # Display table
        if show_reasoning and 'Detailed Reasoning' in display_df.columns:
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Trainset ID": st.column_config.TextColumn("Train ID", width="small"),
                    "Assigned Status": st.column_config.TextColumn(
                        "Status",
                        width="small",
                        help="Current assignment status"
                    ),
                    "Detailed Reasoning": st.column_config.TextColumn("Reasoning", width="large")
                }
            )
        else:
            # Simplified view
            st.dataframe(
                display_df[['Trainset ID', 'Assigned Status']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Trainset ID": st.column_config.TextColumn("Train ID"),
                    "Assigned Status": st.column_config.TextColumn("Status")
                }
            )
        
        # Summary stats
        st.markdown(f"**Showing {len(display_df)} of {len(solution_df)} trains**")
    
    with tab4:
        st.markdown("### ⚠️ Operational Alerts & Notifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🚨 Critical Issues")
            
            critical_count = 0
            
            # Check for certificate issues
            if 'certificates' in data_frames:
                expired_certs = data_frames['certificates'][
                    pd.to_datetime(data_frames['certificates']['expiry_date']).dt.date < datetime.now().date()
                ]
                if not expired_certs.empty:
                    for _, cert in expired_certs.iterrows():
                        st.error(f"❌ {cert['trainset_id']}: {cert['certificate_type']} certificate expired")
                        critical_count += 1
            
            # Check for critical job cards
            if 'job_cards' in data_frames:
                critical_jobs = data_frames['job_cards'][
                    (data_frames['job_cards']['status'] == 'OPEN') & 
                    (data_frames['job_cards']['is_critical'] == True)
                ]
                if not critical_jobs.empty:
                    unique_trains = critical_jobs['trainset_id'].unique()
                    for train in unique_trains[:5]:  # Show first 5
                        job_count = len(critical_jobs[critical_jobs['trainset_id'] == train])
                        st.warning(f"⚠️ {train}: {job_count} critical job(s) pending")
                        critical_count += 1
            
            if critical_count == 0:
                st.success("✅ No critical issues detected")
        
        with col2:
            st.markdown("#### 📋 Maintenance Queue")
            
            maintenance_trains = solution_df[solution_df['Assigned Status'] == 'Maintenance']['Trainset ID'].tolist()
            if maintenance_trains:
                for i, train in enumerate(maintenance_trains[:5], 1):  # Show top 5
                    if 'job_cards' in data_frames:
                        jobs = data_frames['job_cards'][
                            (data_frames['job_cards']['trainset_id'] == train) & 
                            (data_frames['job_cards']['status'] == 'OPEN')
                        ]
                        hours = jobs['required_man_hours'].sum() if not jobs.empty else 0
                        st.info(f"{i}. {train} - Est. {hours:.0f}h work")
                    else:
                        st.info(f"{i}. {train}")
                
                if len(maintenance_trains) > 5:
                    st.caption(f"... and {len(maintenance_trains) - 5} more")
            else:
                st.success("✅ No trains in maintenance queue")
    
    with tab5:
        st.markdown("### 📋 Reports & Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Schedule Export")
            csv = solution_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Schedule (CSV)",
                data=csv,
                file_name=f"train_schedule_{st.session_state.scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.markdown("##### Analytics Report")
            if st.button("📊 Generate Analytics Report", use_container_width=True):
                st.info("Feature coming soon: Comprehensive PDF report with charts")
        
        with col3:
            st.markdown("##### Email Notification")
            if st.button("📧 Send Email Report", use_container_width=True):
                st.info("Feature coming soon: Automated email notifications")
        
        st.markdown("---")
        
        # Summary Statistics Table
        st.markdown("### 📈 Executive Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            summary_data = {
                "Metric": [
                    "Total Fleet Size",
                    "Operational Trains",
                    "Maintenance Backlog",
                    "Average Utilization"
                ],
                "Value": [
                    f"{total_trains}",
                    f"{in_service + standby}",
                    f"{maintenance}",
                    f"{((in_service + standby)/total_trains)*100:.1f}%" if total_trains > 0 else "0.0%"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        with col2:
            if 'trainsets' in data_frames:
                mileage_stats = {
                    "Statistic": [
                        "Min Mileage",
                        "Max Mileage", 
                        "Average Mileage",
                        "Std Deviation"
                    ],
                    "Value (km)": [
                        f"{data_frames['trainsets']['cumulative_mileage_km'].min():,.0f}",
                        f"{data_frames['trainsets']['cumulative_mileage_km'].max():,.0f}",
                        f"{data_frames['trainsets']['cumulative_mileage_km'].mean():,.0f}",
                        f"{data_frames['trainsets']['cumulative_mileage_km'].std():,.0f}"
                    ]
                }
                mileage_df = pd.DataFrame(mileage_stats)
                st.dataframe(mileage_df, use_container_width=True, hide_index=True)

else:
    # Enhanced landing page
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <h2 style="color: #6366f1; font-size: 2.2rem; margin-bottom: 1rem;">Welcome to KMRL AI Train Scheduler</h2>
        <p style="font-size: 1.15rem; color: #64748b; margin: 2rem auto; max-width: 600px;">
            Optimize your fleet operations with AI-powered scheduling. Select a scenario and click 'Optimize' to generate an intelligent train induction plan.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
            <h4 style="color: #6366f1;">Optimized Scheduling</h4>
            <p style="color: #64748b;">AI-driven decisions for maximum fleet efficiency</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <h4 style="color: #6366f1;">Data-Driven Insights</h4>
            <p style="color: #64748b;">Interactive charts and analytics for better planning</p>
        </div>
        """, unsafe_allow_html=True)

    # --- THIS IS THE COMPLETED SECTION ---
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
            <h4 style="color: #6366f1;">Proactive Alerts</h4>
            <p style="color: #64748b;">Monitor critical issues like expired certs and jobs</p>
        </div>
        """, unsafe_allow_html=True)
    # --- END OF COMPLETED SECTION ---