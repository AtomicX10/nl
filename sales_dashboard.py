import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Advanced Sales Dashboard", layout="wide")

st.title("📊 Advanced Sales Analytics Dashboard")

# --------------------------
# Upload dataset
# --------------------------

uploaded_file = st.file_uploader(
    "Upload Sales CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Convert date column
    df["Date"] = pd.to_datetime(df["Date"])

    # --------------------------
    # Sidebar Filters
    # --------------------------

    st.sidebar.header("🔍 Filters")

    region_filter = st.sidebar.multiselect(
        "Select Region",
        df["Region"].unique(),
        default=df["Region"].unique()
    )

    category_filter = st.sidebar.multiselect(
        "Select Category",
        df["Category"].unique(),
        default=df["Category"].unique()
    )

    date_range = st.sidebar.date_input(
        "Select Date Range",
        [df["Date"].min(), df["Date"].max()]
    )

    # Apply filters
    filtered_df = df[
        (df["Region"].isin(region_filter)) &
        (df["Category"].isin(category_filter)) &
        (df["Date"] >= pd.to_datetime(date_range[0])) &
        (df["Date"] <= pd.to_datetime(date_range[1]))
    ]

    # --------------------------
    # KPI Metrics
    # --------------------------

    total_revenue = filtered_df["Revenue"].sum()
    total_profit = filtered_df["Profit"].sum()
    total_quantity = filtered_df["Quantity"].sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Total Revenue", f"{total_revenue:,}")
    col2.metric("📈 Total Profit", f"{total_profit:,}")
    col3.metric("📦 Units Sold", total_quantity)

    st.divider()

    # --------------------------
    # Revenue Trend
    # --------------------------

    st.subheader("📅 Revenue Over Time")

    daily_sales = filtered_df.groupby("Date")["Revenue"].sum()

    fig1, ax1 = plt.subplots()
    ax1.plot(daily_sales.index, daily_sales.values, marker="o")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Revenue")
    ax1.set_title("Daily Revenue Trend")

    st.pyplot(fig1)

    # --------------------------
    # Category Comparison
    # --------------------------

    st.subheader("🛒 Revenue by Category")

    cat_sales = filtered_df.groupby("Category")["Revenue"].sum()

    fig2, ax2 = plt.subplots()
    ax2.bar(cat_sales.index, cat_sales.values)
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Revenue")
    ax2.set_title("Category Revenue Comparison")

    st.pyplot(fig2)

    # --------------------------
    # Region Profit Analysis
    # --------------------------

    st.subheader("🌍 Profit by Region")

    region_profit = filtered_df.groupby("Region")["Profit"].sum()

    fig3, ax3 = plt.subplots()
    ax3.pie(region_profit.values, labels=region_profit.index, autopct="%1.1f%%")
    ax3.set_title("Profit Distribution")

    st.pyplot(fig3)

    # --------------------------
    # Data Preview
    # --------------------------

    st.subheader("📄 Filtered Data")
    st.dataframe(filtered_df)

    st.success("Dashboard Updated Successfully 🚀")

else:
    st.info("Upload a CSV file to begin analysis.")
