#!/usr/bin/env python3
"""Streamlit app to visualize training metrics from metrics.jsonl files."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Training Metrics Viewer", layout="wide")


def load_metrics(path: str) -> pd.DataFrame:
    """Load metrics from a jsonl file into a DataFrame."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def categorize_metrics(columns: list[str]) -> dict[str, list[str]]:
    """Group metrics by their prefix category."""
    categories: dict[str, list[str]] = {}
    for col in columns:
        if col == "step":
            continue
        if "/" in col:
            cat = col.split("/")[0]
        else:
            cat = "other"
        categories.setdefault(cat, []).append(col)
    return categories


def main():
    st.title("Training Metrics Viewer")

    # Sidebar: file selection
    st.sidebar.header("Data Source")
    default_path = "logs/run-5-supervised/metrics.jsonl"
    metrics_path = st.sidebar.text_input("Metrics file path", value=default_path)

    if not Path(metrics_path).exists():
        st.error(f"File not found: {metrics_path}")
        return

    # Load data
    df = load_metrics(metrics_path)
    st.sidebar.success(f"Loaded {len(df)} steps")

    # Categorize metrics
    categories = categorize_metrics(df.columns.tolist())

    # Sidebar: metric selection
    st.sidebar.header("Metrics")
    selected_metrics: list[str] = []

    for cat, metrics in sorted(categories.items()):
        with st.sidebar.expander(f"{cat} ({len(metrics)})", expanded=(cat in ["other", "optim"])):
            for metric in metrics:
                if st.checkbox(metric, value=(metric == "teacher_kl"), key=metric):
                    selected_metrics.append(metric)

    # Main area: charts
    if not selected_metrics:
        st.info("Select metrics from the sidebar to plot")
        return

    # Plot each selected metric
    for metric in selected_metrics:
        st.subheader(metric)
        chart_data = df[["step", metric]].set_index("step")
        st.line_chart(chart_data)

    # Show raw data
    with st.expander("Raw data"):
        st.dataframe(df[["step"] + selected_metrics])


if __name__ == "__main__":
    main()
