"""
Styling utilities for the Delivery Performance Dashboard.
"""
import streamlit as st


def inject_css():
    """
    Inject CSS for KPI cards and dashboard styling.
    Styles KPI cards with rounded corners, subtle shadow, nice spacing, and light background.
    """
    st.markdown(
        """
        <style>
        .kpi-card {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            padding: 1.25rem 1.5rem;
            margin: 0.5rem 0;
            border: 1px solid #e2e8f0;
        }
        .kpi-card .kpi-label {
            font-size: 0.85rem;
            color: #64748b;
            margin-bottom: 0.25rem;
        }
        .kpi-card .kpi-value {
            font-size: 1.75rem;
            font-weight: 600;
            color: #1e293b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
