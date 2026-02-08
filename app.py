"""
Delivery Performance Dashboard - Streamlit App
"""
from datetime import datetime

import pandas as pd
import streamlit as st

from src.charts import make_delay_hist, make_delay_map, make_top_categories
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.metrics import apply_filters, compute_kpis, group_geo
from src.styles import inject_css
from src.transform import get_fact_orders

st.set_page_config(
    page_title="Delivery Performance Dashboard",
    page_icon="📦",
    layout="wide",
)

inject_css()

st.title("Delivery Performance Dashboard")
st.markdown(
    "This dashboard shows **delivery performance** for Brazilian e-commerce orders. "
    "Use the sidebar filters to explore by date, state, category, and payment type. "
    "Charts and KPIs update automatically based on your selection."
)

@st.cache_data
def load_fact_orders():
    """Load fact orders table with caching. Uses processed parquet if available."""
    return get_fact_orders(
        raw_dir=str(RAW_DATA_DIR),
        processed_dir=str(PROCESSED_DATA_DIR),
        use_cache_file=True,
    )


df = load_fact_orders()

def _get_date_range(df: pd.DataFrame):
    if df.empty:
        return datetime(2016, 1, 1).date(), datetime(2018, 12, 31).date()
    ts = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce").dropna()
    if ts.empty:
        return datetime(2016, 1, 1).date(), datetime(2018, 12, 31).date()
    return ts.min().to_pydatetime().date(), ts.max().to_pydatetime().date()


# Sidebar filters
st.sidebar.header("Filters")

_default_start, _default_end = _get_date_range(df)

if st.sidebar.button("Reset filters"):
    st.session_state.start_date = _default_start
    st.session_state.end_date = _default_end
    st.session_state.states = []
    st.session_state.categories = []
    st.session_state.payment_types = []
    st.session_state.delivered_only = False
    st.rerun()

date_cols = st.sidebar.columns(2)
with date_cols[0]:
    start_date = st.date_input("Start date", value=_default_start, key="start_date")
with date_cols[1]:
    end_date = st.date_input("End date", value=_default_end, key="end_date")

date_range = (datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.min.time()))

states = st.sidebar.multiselect(
    "State",
    options=sorted(df["customer_state"].dropna().unique().tolist()) if not df.empty else [],
    default=[],
    key="states",
)

categories = st.sidebar.multiselect(
    "Category",
    options=sorted(df["product_category_mode"].dropna().unique().tolist()) if not df.empty else [],
    default=[],
    key="categories",
)

payment_types = st.sidebar.multiselect(
    "Payment type",
    options=sorted(df["payment_type_mode"].dropna().unique().tolist()) if not df.empty else [],
    default=[],
    key="payment_types",
)

delivered_only = st.sidebar.checkbox(
    "Delivered orders only",
    value=False,
    key="delivered_only",
)

# Apply filters
df_filtered = apply_filters(
    df,
    date_range=date_range,
    states=states if states else None,
    categories=categories if categories else None,
    payment_types=payment_types if payment_types else None,
    delivered_only=delivered_only,
)

# Empty-state warning
if df_filtered.empty:
    st.warning("No data matches the current filters. Adjust filters or load data.")
else:
    kpis = compute_kpis(df_filtered)

    # KPI row with 4 metrics (tooltips via title attribute)
    kpi_cols = st.columns(4)
    kpi_data = [
        ("Total Orders", lambda: f"{kpis['total_orders']:,}", "Total number of orders in the filtered dataset"),
        ("Delivered", lambda: f"{kpis['delivered_orders']:,}", "Orders with status 'delivered'"),
        ("On-Time Rate", lambda: f"{kpis['on_time_rate']:.1%}", "Share of delivered orders that arrived on or before estimated date"),
        ("Avg Delay (days)", lambda: f"{kpis['avg_delay_days']:.1f}", "Average delay in days (negative = early delivery)"),
    ]
    for col, (label, fmt, tooltip) in zip(kpi_cols, kpi_data):
        with col:
            st.markdown(
                f'<div class="kpi-card" title="{tooltip}">'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{fmt()}</div></div>',
                unsafe_allow_html=True,
            )

    # Map + histogram row
    map_col, hist_col = st.columns(2)

    df_geo = group_geo(df_filtered)
    # Drop NaN lat/lng before map (group_geo already does this; ensure clean pass)
    df_geo_map = df_geo.dropna(subset=["mean_lat", "mean_lng"]) if not df_geo.empty else df_geo

    with map_col:
        if df_geo_map.empty:
            st.warning("No geographic data available for map. Rows with missing lat/lng were dropped.")
        fig_map = make_delay_map(df_geo_map)
        st.plotly_chart(fig_map, width="stretch")

    with hist_col:
        fig_hist = make_delay_hist(df_filtered)
        st.plotly_chart(fig_hist, width="stretch")

    # Tabs: Root causes and Data table
    tab1, tab2 = st.tabs(["Root Causes", "Data Table"])

    with tab1:
        fig_cats = make_top_categories(df_filtered)
        st.plotly_chart(fig_cats, width="stretch")

    with tab2:
        st.dataframe(df_filtered, use_container_width=True)

        # Download button
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="fact_orders.csv",
            mime="text/csv",
        )
