"""Streamlit dashboard for POC rally analytics output.

Run after poc.py has generated poc_output/rally_summary.json:
    streamlit run rally_dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SUMMARY_PATH = Path("poc_output/rally_summary.json")
HEATMAP_DIR = Path("poc_output")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Rally Analytics", layout="wide")
st.title("Rally Analytics Dashboard")

if not SUMMARY_PATH.exists():
    st.error(
        f"`{SUMMARY_PATH}` not found. Run `python poc.py` first to generate it."
    )
    st.stop()

with open(SUMMARY_PATH) as f:
    rallies = json.load(f)

if not rallies:
    st.warning("No rallies detected in the summary file.")
    st.stop()

# Build flat DataFrames
rally_df = pd.DataFrame([
    {
        "rally": i + 1,
        "start_time": r["start_time"],
        "end_time": r["end_time"],
        "duration_s": r["duration_s"],
        "shot_count": r["shot_count"],
        "max_shot_speed_kmh": r["max_shot_speed_kmh"],
    }
    for i, r in enumerate(rallies)
])

shots_rows = []
for i, r in enumerate(rallies):
    for s in r["shots"]:
        shots_rows.append({
            "rally": i + 1,
            "timestamp": s["timestamp"],
            "speed_kmh": s["speed_kmh"],
        })
shots_df = pd.DataFrame(shots_rows) if shots_rows else pd.DataFrame()

# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------

all_speeds = shots_df["speed_kmh"].dropna().tolist() if not shots_df.empty else []

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Rallies", len(rallies))
col2.metric("Total rally time", f"{rally_df['duration_s'].sum():.0f}s")
col3.metric("Total shots", int(rally_df["shot_count"].sum()))
col4.metric("Max shot speed", f"{max(all_speeds):.0f} km/h" if all_speeds else "N/A")
col5.metric("Avg rally duration", f"{rally_df['duration_s'].mean():.1f}s")

st.divider()

# ---------------------------------------------------------------------------
# Row 1: Duration bar + shot count bar
# ---------------------------------------------------------------------------

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Rally Duration")
    fig = go.Figure(go.Bar(
        x=rally_df["rally"],
        y=rally_df["duration_s"],
        marker_color=rally_df["shot_count"],
        marker_colorscale="Viridis",
        marker_showscale=True,
        marker_colorbar_title="Shots",
        text=rally_df["duration_s"].map(lambda v: f"{v:.1f}s"),
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Rally #",
        yaxis_title="Duration (s)",
        margin=dict(t=20, b=20),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Shot Count per Rally")
    fig = go.Figure(go.Bar(
        x=rally_df["rally"],
        y=rally_df["shot_count"],
        marker_color=rally_df["shot_count"],
        marker_colorscale="Blues",
        text=rally_df["shot_count"],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Rally #",
        yaxis_title="Shots",
        margin=dict(t=20, b=20),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 2: Timeline + shot speed scatter
# ---------------------------------------------------------------------------

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Rally Timeline")
    fig = go.Figure()
    for _, row in rally_df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["duration_s"]],
            y=[f"Rally {int(row['rally'])}"],
            base=[row["start_time"]],
            orientation="h",
            marker_color=row["shot_count"],
            marker_colorscale="Viridis",
            showlegend=False,
            hovertemplate=(
                f"Rally {int(row['rally'])}<br>"
                f"Start: {row['start_time']:.1f}s<br>"
                f"End: {row['end_time']:.1f}s<br>"
                f"Duration: {row['duration_s']:.1f}s<br>"
                f"Shots: {int(row['shot_count'])}<extra></extra>"
            ),
        ))
    fig.update_layout(
        xaxis_title="Time (s)",
        barmode="overlay",
        margin=dict(t=20, b=20),
        height=max(250, len(rallies) * 28),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_d:
    st.subheader("Shot Speeds over Time")
    if not shots_df.empty and shots_df["speed_kmh"].notna().any():
        fig = go.Figure(go.Scatter(
            x=shots_df["timestamp"],
            y=shots_df["speed_kmh"],
            mode="markers",
            marker=dict(
                color=shots_df["speed_kmh"],
                colorscale="Jet",
                size=8,
                showscale=True,
                colorbar_title="km/h",
            ),
            text=shots_df["rally"].map(lambda r: f"Rally {r}"),
            hovertemplate="%{text}<br>Time: %{x:.1f}s<br>Speed: %{y:.1f} km/h<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Speed (km/h)",
            margin=dict(t=20, b=20),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No shot speed data available (requires projected court coordinates).")

# ---------------------------------------------------------------------------
# Max shot speed per rally
# ---------------------------------------------------------------------------

if rally_df["max_shot_speed_kmh"].notna().any():
    st.subheader("Max Shot Speed per Rally")
    fig = go.Figure(go.Bar(
        x=rally_df["rally"],
        y=rally_df["max_shot_speed_kmh"],
        marker_color=rally_df["max_shot_speed_kmh"],
        marker_colorscale="Reds",
        text=rally_df["max_shot_speed_kmh"].map(
            lambda v: f"{v:.0f}" if pd.notna(v) else ""
        ),
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Rally #",
        yaxis_title="Max speed (km/h)",
        margin=dict(t=20, b=20),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Rally detail drill-down
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Rally Detail")

rally_choice = st.selectbox(
    "Select a rally",
    options=list(range(1, len(rallies) + 1)),
    format_func=lambda i: (
        f"Rally {i}  –  {rallies[i-1]['duration_s']:.1f}s  "
        f"({rallies[i-1]['shot_count']} shots)"
    ),
)

r = rallies[rally_choice - 1]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Start", f"{r['start_time']:.1f}s")
m2.metric("End", f"{r['end_time']:.1f}s")
m3.metric("Duration", f"{r['duration_s']:.1f}s")
m4.metric("Shots", r["shot_count"])

if r["shots"]:
    shot_detail_df = pd.DataFrame(r["shots"])
    shot_detail_df.index += 1
    shot_detail_df.columns = ["Frame", "Timestamp (s)", "Speed (km/h)"]
    st.dataframe(shot_detail_df, use_container_width=True)

    if shot_detail_df["Speed (km/h)"].notna().any():
        fig = go.Figure(go.Scatter(
            x=shot_detail_df["Timestamp (s)"],
            y=shot_detail_df["Speed (km/h)"],
            mode="lines+markers",
            marker=dict(size=8, color="orange"),
            line=dict(color="orange"),
        ))
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Speed (km/h)",
            title=f"Shot speeds in Rally {rally_choice}",
            height=280,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No shots detected in this rally.")

# ---------------------------------------------------------------------------
# Rally summary table
# ---------------------------------------------------------------------------

st.divider()
st.subheader("All Rallies")
display_df = rally_df.copy()
display_df.columns = [
    "Rally #", "Start (s)", "End (s)", "Duration (s)", "Shots", "Max Speed (km/h)"
]
display_df = display_df.set_index("Rally #")
st.dataframe(display_df.style.background_gradient(subset=["Shots"], cmap="Blues"), use_container_width=True)

# ---------------------------------------------------------------------------
# Player heatmaps
# ---------------------------------------------------------------------------

heatmap_files = sorted(HEATMAP_DIR.glob("heatmap_player*.png"))
if heatmap_files:
    st.divider()
    st.subheader("Player Heatmaps")
    cols = st.columns(len(heatmap_files))
    for col, path in zip(cols, heatmap_files):
        with col:
            st.image(str(path), caption=path.stem.replace("_", " ").title())
