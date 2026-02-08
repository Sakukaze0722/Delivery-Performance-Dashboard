"""
Chart and visualization components for the Delivery Performance Dashboard.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _empty_map_figure(title: str, message: str):
    """Return an empty map figure with message for no-data / error states."""
    empty_df = pd.DataFrame({"mean_lat": [-15], "mean_lng": [-50], "label": [message]})
    fig = px.scatter_mapbox(
        empty_df, lat="mean_lat", lon="mean_lng",
        mapbox_style="open-street-map", zoom=3,
        center={"lat": -15, "lon": -50},
    )
    fig.update_traces(marker={"size": 0, "opacity": 0})
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(title=title, margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig


def make_delay_map(df_geo: pd.DataFrame):
    """
    Create a scatter mapbox showing avg_delay_days by region.
    Uses OpenStreetMap style.
    Returns a Plotly figure with clean title and hover info.
    """
    if df_geo is None or df_geo.empty:
        return _empty_map_figure("Delivery Delay by Region (Map)", "No data to display")

    required = ["mean_lat", "mean_lng"]
    for col in required:
        if col not in df_geo.columns:
            return _empty_map_figure("Delivery Delay by Region (Map)", "Missing geographic data")

    plot_df = df_geo.dropna(subset=["mean_lat", "mean_lng"])
    if plot_df.empty:
        return _empty_map_figure("Delivery Delay by Region (Map)", "No valid coordinates")

    fig = px.scatter_mapbox(
        plot_df,
        lat="mean_lat",
        lon="mean_lng",
        color="avg_delay_days" if "avg_delay_days" in plot_df.columns else None,
        size="order_count" if "order_count" in plot_df.columns else None,
        hover_name="customer_state" if "customer_state" in plot_df.columns else None,
        hover_data={
            "mean_lat": ":.4f",
            "mean_lng": ":.4f",
            "order_count": True,
            "delivered_count": True,
            "on_time_rate": ":.2%",
            "avg_delay_days": ":.1f",
        },
        mapbox_style="open-street-map",
        zoom=3,
        center={"lat": plot_df["mean_lat"].mean(), "lon": plot_df["mean_lng"].mean()},
    )
    fig.update_layout(
        title="Delivery Delay by Region (negative = early)",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )
    return fig


def _empty_figure(title: str, message: str):
    """Return an empty figure with message for no-data / error states."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
    )
    fig.update_layout(title=title)
    return fig


def make_delay_hist(df: pd.DataFrame):
    """
    Create a histogram of delay_days for delivered orders.
    Returns a Plotly figure with clean title and hover info.
    """
    if df is None or df.empty:
        return _empty_figure("Distribution of Delivery Delay (days)", "No data to display")

    if "delay_days" not in df.columns:
        return _empty_figure("Distribution of Delivery Delay (days)", "Missing delay_days column")

    plot_df = df[["delay_days"]].dropna()
    if plot_df.empty:
        return _empty_figure("Distribution of Delivery Delay (days)", "No delay data available")

    fig = px.histogram(
        plot_df,
        x="delay_days",
        nbins=50,
        labels={"delay_days": "Delay (days)", "count": "Orders"},
        title="Distribution of Delivery Delay (days)",
    )
    fig.update_traces(
        hovertemplate="Delay: %{x} days<br>Orders: %{y}<extra></extra>"
    )
    fig.update_layout(
        xaxis_title="Delay (days)",
        yaxis_title="Number of Orders",
        showlegend=False,
    )
    return fig


def make_top_categories(df: pd.DataFrame):
    """
    Create a bar chart of top 10 worst on_time_rate categories.
    Returns a Plotly figure with clean title and hover info.
    """
    title = "Worst On-Time Rate by Category (Top 10)"
    if df is None or df.empty:
        return _empty_figure(title, "No data to display")

    if "product_category_mode" not in df.columns or "on_time" not in df.columns:
        return _empty_figure(title, "Missing category or on_time data")

    delivered = df[df["order_status"] == "delivered"]
    if delivered.empty:
        return _empty_figure(title, "No delivered orders")

    agg = (
        delivered.groupby("product_category_mode")
        .agg(
            on_time_count=("on_time", lambda x: (x == True).sum()),
            total=("order_id", "nunique"),
        )
        .reset_index()
    )
    agg["on_time_rate"] = agg["on_time_count"] / agg["total"].replace(0, pd.NA)
    agg = agg.dropna(subset=["on_time_rate"])
    agg = agg.nsmallest(10, "on_time_rate")
    if agg.empty:
        return _empty_figure(title, "No categories with sufficient data")

    fig = px.bar(
        agg,
        x="product_category_mode",
        y="on_time_rate",
        labels={
            "product_category_mode": "Category",
            "on_time_rate": "On-Time Rate",
        },
        title="Worst On-Time Rate by Category (Top 10)",
        hover_data={"total": True, "on_time_count": True},
    )
    fig.update_traces(
        hovertemplate="Category: %{x}<br>On-Time Rate: %{y:.1%}<br>Orders: %{customdata[0]}<extra></extra>"
    )
    fig.update_layout(
        xaxis_title="Product Category",
        yaxis_title="On-Time Rate",
        xaxis_tickangle=-45,
        yaxis_tickformat=".0%",
    )
    return fig
