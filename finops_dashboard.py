"""
FinOps Cloud Cost Analytics Dashboard

A Streamlit application for analyzing simulated cloud spending data,
identifying optimization opportunities, and forecasting future costs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def load_data(file_path: str = "simulated_cur_report.csv") -> pd.DataFrame:
    """Load and preprocess the simulated CUR data."""
    df = pd.read_csv(file_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Date"] = df["Timestamp"].dt.date
    return df


def get_top_services(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Get the top N most expensive services by total cost."""
    return (
        df.groupby("Service")["Unblended_Cost"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )


def get_expensive_resources(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Identify the top N most expensive individual resources (potential zombies)."""
    return (
        df.groupby(["Service", "Resource_ID"])["Unblended_Cost"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )


def forecast_monthly_cost(df: pd.DataFrame) -> dict:
    """
    Perform simple linear regression on daily costs to forecast monthly spending.
    
    Returns a dictionary with:
    - daily_costs: DataFrame of daily aggregated costs
    - slope: Daily cost trend
    - intercept: Baseline cost
    - projected_monthly: Projected total monthly cost
    - current_total: Current total cost in the data
    """
    # Aggregate costs by date
    daily_costs = (
        df.groupby("Date")["Unblended_Cost"]
        .sum()
        .reset_index()
    )
    daily_costs["Day_Number"] = range(len(daily_costs))
    
    # Simple linear regression using numpy
    x = daily_costs["Day_Number"].values
    y = daily_costs["Unblended_Cost"].values
    
    # Calculate slope and intercept: y = mx + b
    n = len(x)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    intercept = (np.sum(y) - slope * np.sum(x)) / n
    
    # Project for 30 days (full month)
    projected_daily_costs = [slope * day + intercept for day in range(30)]
    projected_monthly = sum(projected_daily_costs)
    
    # Calculate trend line for visualization
    daily_costs["Trend"] = slope * daily_costs["Day_Number"] + intercept
    
    # Flag if the forecast is unrealistic (negative projection)
    is_unrealistic = projected_monthly < 0
    
    return {
        "daily_costs": daily_costs,
        "slope": slope,
        "intercept": intercept,
        "projected_monthly": max(0, projected_monthly),
        "current_total": df["Unblended_Cost"].sum(),
        "avg_daily_cost": y.mean(),
        "is_unrealistic_forecast": is_unrealistic
    }


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="FinOps Cloud Cost Dashboard",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 FinOps Cloud Cost Optimization Dashboard")
    st.markdown("""
    This dashboard analyzes simulated cloud spending data to identify cost optimization 
    opportunities and forecast future expenses.
    """)
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("⚠️ Data file 'simulated_cur_report.csv' not found. Please run `python generate_cur_data.py` first.")
        st.stop()
    
    # Display key metrics
    st.header("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cost", f"${df['Unblended_Cost'].sum():,.2f}")
    with col2:
        st.metric("Total Services", df["Service"].nunique())
    with col3:
        st.metric("Total Resources", df["Resource_ID"].nunique())
    with col4:
        st.metric("Billing Records", len(df))
    
    st.divider()
    
    # Analysis 1: Cost Aggregation - Top 5 Services
    st.header("📈 Analysis 1: Top 5 Most Expensive Services")
    
    top_services = get_top_services(df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(top_services)))
        bars = ax.barh(top_services["Service"], top_services["Unblended_Cost"], color=colors)
        ax.set_xlabel("Total Cost ($)")
        ax.set_ylabel("Service")
        ax.set_title("Top 5 Most Expensive Cloud Services")
        ax.invert_yaxis()
        
        # Add value labels on bars
        for bar, cost in zip(bars, top_services["Unblended_Cost"]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                   f"${cost:,.2f}", va="center", fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Cost Breakdown")
        for _, row in top_services.iterrows():
            pct = row["Unblended_Cost"] / df["Unblended_Cost"].sum() * 100
            st.write(f"**{row['Service']}**: ${row['Unblended_Cost']:,.2f} ({pct:.1f}%)")
    
    st.divider()
    
    # Analysis 2: Optimization Flag - Expensive Resources
    st.header("🚨 Analysis 2: Top 3 Most Expensive Resources (Potential Zombies)")
    st.markdown("""
    These resources have the highest individual costs and may represent:
    - **Zombie resources**: Unused but still running
    - **Misconfigured instances**: Over-provisioned or inefficient configurations
    - **Optimization opportunities**: Candidates for right-sizing or termination
    """)
    
    expensive_resources = get_expensive_resources(df)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        for idx, row in expensive_resources.iterrows():
            st.error(f"""
            **#{idx + 1} - {row['Resource_ID']}**
            - Service: {row['Service']}
            - Total Cost: ${row['Unblended_Cost']:,.2f}
            """)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#ff6b6b", "#ffa502", "#ff9ff3"]
        ax.bar(expensive_resources["Resource_ID"], expensive_resources["Unblended_Cost"], color=colors)
        ax.set_xlabel("Resource ID")
        ax.set_ylabel("Total Cost ($)")
        ax.set_title("Top 3 Most Expensive Individual Resources")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
    
    st.divider()
    
    # Analysis 3: Simple Forecasting
    st.header("🔮 Analysis 3: Monthly Cost Forecast")
    
    forecast = forecast_monthly_cost(df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        daily_costs = forecast["daily_costs"]
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Plot actual daily costs
        ax.plot(daily_costs["Date"], daily_costs["Unblended_Cost"], 
               marker="o", linestyle="-", color="#3498db", label="Actual Daily Cost", markersize=4)
        
        # Plot trend line
        ax.plot(daily_costs["Date"], daily_costs["Trend"], 
               linestyle="--", color="#e74c3c", label="Trend Line", linewidth=2)
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Cost ($)")
        ax.set_title("Daily Cost Trend with Linear Regression Forecast")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Forecast Summary")
        st.metric("Current Period Total", f"${forecast['current_total']:,.2f}")
        st.metric("Average Daily Cost", f"${forecast['avg_daily_cost']:,.2f}")
        st.metric("Projected Monthly Cost", f"${forecast['projected_monthly']:,.2f}")
        
        trend_direction = "📈 Increasing" if forecast["slope"] > 0 else "📉 Decreasing"
        st.info(f"**Trend**: {trend_direction}\n\nDaily change: ${abs(forecast['slope']):.2f}")
        
        if forecast.get("is_unrealistic_forecast"):
            st.warning("⚠️ The linear model produced an unrealistic negative projection. "
                      "The displayed value has been adjusted to $0. Consider using a "
                      "more sophisticated forecasting model for better accuracy.")
    
    st.divider()
    
    # Raw Data View
    st.header("📋 Raw Data Explorer")
    
    with st.expander("View Raw Billing Data"):
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="cloud_cost_data.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
