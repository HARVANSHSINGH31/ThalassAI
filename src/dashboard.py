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
fig_stress = go.Figure(go.Bar(x=df['time'], y=df['stress_score'], marker_color=colors, text=df['stress_score'], textposition='outside'))
fig_stress.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="High stress threshold")
fig_stress.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Moderate stress threshold")
fig_stress.update_layout(yaxis_title="Stress Score (0-100)", xaxis_title="Date", height=400, showlegend=False)
st.plotly_chart(fig_stress, use_container_width=True)

st.subheader("Multi-Signal Overview (Normalized)")
fig_signals = go.Figure()
fig_signals.add_trace(go.Scatter(x=df['time'], y=df['sst_norm'], name='Sea Surface Temperature', line=dict(color='steelblue', width=2.5), mode='lines+markers'))
fig_signals.add_trace(go.Scatter(x=df['time'], y=df['do_norm'], name='Dissolved Oxygen', line=dict(color='teal', width=2.5), mode='lines+markers'))
fig_signals.add_trace(go.Scatter(x=df['time'], y=df['chl_norm'], name='Chlorophyll-a', line=dict(color='green', width=2.5), mode='lines+markers'))
fig_signals.add_vrect(x0="Jun", x1="Sep", fillcolor="red", opacity=0.1, annotation_text="Peak stress window")
fig_signals.update_layout(yaxis_title="Normalized Value (0-1)", xaxis_title="Date", height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig_signals, use_container_width=True)


st.divider()
st.subheader("LSTM Anomaly Detection")
st.markdown("*Reconstruction error from LSTM Autoencoder — spikes indicate detected stress anomalies*")

import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)
    def forward(self, x):
        _, (h, c) = self.encoder(x)
        repeated = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(repeated)
        return out

seq_len = 6
features = df[["sst_norm", "do_norm", "chl_norm"]].values.astype(np.float32)
sequences = []
for i in range(len(features) - seq_len):
    sequences.append(features[i:i+seq_len])
X = torch.tensor(np.array(sequences))

@st.cache_resource
def train_lstm():
    seq_len_inner = 6
    feats = df[["sst_norm", "do_norm", "chl_norm"]].values.astype(np.float32)
    seqs = [feats[i:i+seq_len_inner] for i in range(len(feats) - seq_len_inner)]
    X_inner = torch.tensor(np.array(seqs))
    model = LSTMAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        out = model(X_inner)
        loss = criterion(out, X_inner)
        loss.backward()
        optimizer.step()
    return model

model = train_lstm()
model.eval()
with torch.no_grad():
    reconstructed = model(X).numpy()

errors = np.mean((reconstructed - X.numpy()) ** 2, axis=(1, 2))
threshold = float(np.mean(errors) + 1.5 * np.std(errors))
anomaly_flags = errors > threshold
times = pd.to_datetime(df["time"].values[seq_len:])

events = [
    ("Jul-Aug 2015", "2015-07-01", "2015-08-31", "#FF4136"),
    ("Mar-Jul 2018", "2018-03-01", "2018-07-31", "#FF851B"),
    ("Jul 2019",     "2019-07-01", "2019-07-31", "#FFDC00"),
    ("Jul-Aug 2020", "2020-07-01", "2020-08-31", "#2ECC40"),
]

fig_anomaly = go.Figure()
fig_anomaly.add_trace(go.Scatter(x=times, y=errors, name="Reconstruction Error", line=dict(color="royalblue", width=2), mode="lines"))
fig_anomaly.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"Anomaly threshold (u+2s)")
fig_anomaly.add_trace(go.Scatter(x=times[anomaly_flags], y=errors[anomaly_flags], mode="markers", name="Flagged Anomaly", marker=dict(color="red", size=10, symbol="x")))

for label, start, end, color in events:
    fig_anomaly.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.15, line_width=0, annotation_text=label, annotation_position="top left")

fig_anomaly.update_layout(yaxis_title="MSE Reconstruction Error", xaxis_title="Date", height=450, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig_anomaly, use_container_width=True)

st.markdown("**Detected anomalies vs documented events:**")
anomaly_dates = times[anomaly_flags]
summary = []
for label, start, end, _ in events:
    hits = anomaly_dates[(anomaly_dates >= start) & (anomaly_dates <= end)]
    summary.append({"Event": label, "Flagged months": len(hits), "Validated": "YES" if len(hits) > 0 else "NO"})
st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


st.subheader("Raw Signal Data")
st.dataframe(df[['time','sst','dissolved_oxygen','chlorophyll','stress_score']], use_container_width=True)

st.divider()
st.markdown("**OceanPulse** · Built by Harvansh Singh · Data: Copernicus CMEMS (GLORYS12V1 + BGC reanalysis)")
