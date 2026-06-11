import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title="OceanPulse", page_icon="🌊", layout="wide")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'processed', 'master_dataset_real.csv')
    df = pd.read_csv(csv_path)
    return df

df = load_data()

st.title("🌊 OceanPulse")
st.markdown("**Multi-Signal Ocean Ecosystem Stress Detection System**")
st.markdown("*Arabian Sea · 20°N 65°E · Real oceanographic data*")
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Peak Stress Month", "July", "100/100")
col2.metric("Max Stress Score", "100.0", "July")
col3.metric("Min Dissolved O₂", "3.89 mL/L", "July — stress zone")
col4.metric("Peak Chlorophyll", "0.61 mg/m³", "Monsoon bloom")
st.divider()

st.subheader("Ecosystem Stress Score by Month")
colors = ['#d73027' if s >= 60 else '#fc8d59' if s >= 40 else '#91cf60' for s in df['stress_score']]
fig_stress = go.Figure(go.Bar(x=df['month'], y=df['stress_score'], marker_color=colors, text=df['stress_score'], textposition='outside'))
fig_stress.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="High stress threshold")
fig_stress.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Moderate stress threshold")
fig_stress.update_layout(yaxis_title="Stress Score (0-100)", xaxis_title="Month", height=400, showlegend=False)
st.plotly_chart(fig_stress, use_container_width=True)

st.subheader("Multi-Signal Overview (Normalized)")
fig_signals = go.Figure()
fig_signals.add_trace(go.Scatter(x=df['month'], y=df['sst_norm'], name='Sea Surface Temperature', line=dict(color='steelblue', width=2.5), mode='lines+markers'))
fig_signals.add_trace(go.Scatter(x=df['month'], y=df['do_norm'], name='Dissolved Oxygen', line=dict(color='teal', width=2.5), mode='lines+markers'))
fig_signals.add_trace(go.Scatter(x=df['month'], y=df['chl_norm'], name='Chlorophyll-a', line=dict(color='green', width=2.5), mode='lines+markers'))
fig_signals.add_vrect(x0="Jun", x1="Sep", fillcolor="red", opacity=0.1, annotation_text="Peak stress window")
fig_signals.update_layout(yaxis_title="Normalized Value (0-1)", xaxis_title="Month", height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig_signals, use_container_width=True)

st.subheader("Raw Signal Data")
st.dataframe(df[['month','sst','dissolved_oxygen','chlorophyll','stress_score']], use_container_width=True)

st.divider()
st.markdown("**OceanPulse** · Built by Harvansh Singh · Data: NASA MUR SST, WOA13, MODIS-Aqua")
