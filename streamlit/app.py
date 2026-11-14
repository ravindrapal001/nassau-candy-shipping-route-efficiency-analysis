from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Nassau Candy | Business Intelligence",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        background: #F4F7FB;
        border: 1px solid #DCE5F0;
        padding: 16px 18px;
        border-radius: 12px;
    }
    [data-testid="stMetricLabel"] {
        color: #52657A;
        font-size: 0.88rem;
    }
    [data-testid="stMetricValue"] {
        color: #183153;
        font-weight: 700;
    }
    div[data-testid="stPlotlyChart"] {
        border: 1px solid #E6EBF2;
        border-radius: 12px;
        padding: 6px;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT PATHS AND DATA LOADING
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_DIR.parent / "data" / "raw"

SOURCE_FILES = {
    "FactSales": "FactSales.csv",
    "DimCustomer": "DimCustomer.csv",
    "DimDate": "DimDate.csv",
    "DimGeography": "DimGeography.csv",
    "DimProduct": "DimProduct.csv",
    "DimFactory": "DimFactory.csv",
}


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert source column names to lowercase snake_case."""
    result = df.copy()
    result.columns = [
        re.sub(
            r"_+",
            "_",
            re.sub(r"[^A-Za-z0-9]+", "_", str(col).strip()),
        ).strip("_").lower()
        for col in result.columns
    ]
    return result


@st.cache_data(show_spinner="Loading source data and preparing the analytical model...")
def load_analytical_model(data_folder: str) -> pd.DataFrame:
    """Load six source CSVs, standardize them and build the joined model."""
    folder = Path(data_folder)
    tables = {}

    for table_name, file_name in SOURCE_FILES.items():
        file_path = folder / file_name
        if not file_path.exists():
            raise FileNotFoundError(
                f"Cannot find {file_name}. Expected location: {file_path}"
            )

        tables[table_name] = standardize_column_names(
            pd.read_csv(file_path, low_memory=False)
        )

    # Convert relationship keys to nullable integers.
    key_columns = {
        "FactSales": [
            "row_id", "orderdatekey", "shipdatekey", "customerkey",
            "geographykey", "productkey", "factorykey",
        ],
        "DimCustomer": ["customerkey"],
        "DimDate": ["datekey"],
        "DimGeography": ["geographykey"],
        "DimProduct": ["productkey"],
        "DimFactory": ["factorykey"],
    }

    for table_name, columns in key_columns.items():
        for column in columns:
            tables[table_name][column] = pd.to_numeric(
                tables[table_name][column], errors="raise"
            ).astype("Int64")

    fact = tables["FactSales"].copy()

    for column in ["sales", "units", "gross_profit", "cost"]:
        fact[column] = pd.to_numeric(fact[column], errors="coerce")

    # Join the non-date dimensions.
    joins = [
        ("DimCustomer", "customerkey"),
        ("DimGeography", "geographykey"),
        ("DimProduct", "productkey"),
        ("DimFactory", "factorykey"),
    ]

    model = fact.copy()

    for table_name, key in joins:
        dimension = tables[table_name].copy()
        if dimension[key].duplicated().any():
            raise ValueError(
                f"{table_name}.{key} contains duplicate keys. "
                "Resolve the dimension key issue before joining."
            )
        model = model.merge(
            dimension,
            how="left",
            on=key,
            validate="many_to_one",
        )

    # DimDate is a role-playing dimension: join once for order date,
    # and a second time for ship date.
    date_dimension = tables["DimDate"][["datekey", "date"]].copy()
    date_dimension["date"] = pd.to_datetime(
        date_dimension["date"], errors="coerce"
    )

    if date_dimension["datekey"].duplicated().any():
        raise ValueError("DimDate.datekey contains duplicate keys.")

    order_lookup = date_dimension.rename(
        columns={"datekey": "orderdatekey", "date": "order_date"}
    )
    ship_lookup = date_dimension.rename(
        columns={"datekey": "shipdatekey", "date": "ship_date"}
    )

    model = model.merge(
        order_lookup,
        how="left",
        on="orderdatekey",
        validate="many_to_one",
    )
    model = model.merge(
        ship_lookup,
        how="left",
        on="shipdatekey",
        validate="many_to_one",
    )

    # Calendar fields are derived from Order Date.
    model["order_year"] = model["order_date"].dt.year
    model["order_month"] = model["order_date"].dt.month
    model["order_month_name"] = model["order_date"].dt.month_name()
    model["order_quarter"] = model["order_date"].dt.quarter

    # Route ID means factory + destination combination, not a road route.
    model["route_id"] = (
        model["factory"].fillna("Unknown").astype(str)
        + " → "
        + model["city"].fillna("Unknown").astype(str)
        + ", "
        + model["state_province"].fillna("Unknown").astype(str)
        + ", "
        + model["country_region"].fillna("Unknown").astype(str)
    )

    return model


try:
    model = load_analytical_model(str(RAW_DATA_DIR))
except Exception as exc:
    st.error("The dashboard could not prepare the analytical model.")
    st.exception(exc)
    st.stop()


# ============================================================
# SIDEBAR: NAVIGATION AND FILTERS
# ============================================================

st.sidebar.title("🍬 Nassau Candy")
st.sidebar.caption("Business Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Executive Overview",
        "Factory Analysis",
        "Product Analysis",
        "Geography & Routes",
        "Shipping Analysis",
        "Transaction Explorer",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("GLOBAL FILTERS")

years = sorted(model["order_year"].dropna().astype(int).unique().tolist())
regions = sorted(model["region"].dropna().astype(str).unique().tolist())
factories = sorted(model["factory"].dropna().astype(str).unique().tolist())
ship_modes = sorted(model["ship_mode"].dropna().astype(str).unique().tolist())

selected_years = st.sidebar.multiselect(
    "Order Year", years, default=years
)
selected_regions = st.sidebar.multiselect(
    "Region", regions, default=regions
)
selected_factories = st.sidebar.multiselect(
    "Factory", factories, default=factories
)
selected_ship_modes = st.sidebar.multiselect(
    "Shipping Mode", ship_modes, default=ship_modes
)

valid_order_dates = model["order_date"].dropna()
if valid_order_dates.empty:
    st.error("No valid Order Date values were found in the date dimension.")
    st.stop()

min_date = valid_order_dates.min().date()
max_date = valid_order_dates.max().date()

date_selection = st.sidebar.date_input(
    "Order Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_selection, (tuple, list)):
    if len(date_selection) == 2:
        start_date, end_date = date_selection
    elif len(date_selection) == 1:
        start_date = end_date = date_selection[0]
    else:
        start_date, end_date = min_date, max_date
else:
    start_date = end_date = date_selection

if start_date > end_date:
    start_date, end_date = end_date, start_date

filtered = model[
    model["order_year"].isin(selected_years)
    & model["region"].isin(selected_regions)
    & model["factory"].isin(selected_factories)
    & model["ship_mode"].isin(selected_ship_modes)
    & model["order_date"].dt.date.between(start_date, end_date)
].copy()

st.sidebar.caption(f"Filtered transaction rows: {len(filtered):,}")


# ============================================================
# SHARED HELPERS
# ============================================================

def show_page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def show_kpis(data: pd.DataFrame) -> None:
    sales = data["sales"].sum()
    orders = data["order_id"].nunique()
    units = data["units"].sum()
    gross_profit = data["gross_profit"].sum()
    margin = gross_profit / sales * 100 if sales else 0
    aov = sales / orders if orders else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Sales", f"{sales:,.2f}")
    k2.metric("Distinct Orders", f"{orders:,.0f}")
    k3.metric("Units Sold", f"{units:,.0f}")

    k4, k5, k6 = st.columns(3)
    k4.metric("Gross Profit", f"{gross_profit:,.2f}")
    k5.metric("Gross Profit Margin", f"{margin:.2f}%")
    k6.metric("Average Order Value", f"{aov:,.2f}")


def chart_layout(fig, height=380):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=60, b=25),
        legend_title_text="",
    )
    return fig


if filtered.empty:
    st.warning(
        "No records match the current filters. Select at least one "
        "year, region, factory and shipping mode, and adjust the date range."
    )
    st.stop()


# ============================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":
    show_page_header(
        "Executive Overview",
        "A consolidated view of sales, orders, profitability and business activity.",
    )
    show_kpis(filtered)
    st.divider()

    monthly = (
        filtered.groupby(["order_year", "order_month"], as_index=False)
        .agg(total_sales=("sales", "sum"))
        .sort_values(["order_year", "order_month"])
    )
    monthly["Month"] = (
        monthly["order_year"].astype(str)
        + "-"
        + monthly["order_month"].astype(str).str.zfill(2)
    )

    factory = (
        filtered.groupby("factory", as_index=False)
        .agg(total_sales=("sales", "sum"))
        .sort_values("total_sales", ascending=True)
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(
            monthly, x="Month", y="total_sales", markers=True,
            title="Monthly Sales Trend",
            labels={"total_sales": "Sales", "Month": "Order Month"},
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    with c2:
        fig = px.bar(
            factory, x="total_sales", y="factory", orientation="h",
            title="Sales by Factory",
            labels={"total_sales": "Sales", "factory": "Factory"},
            color="total_sales", color_continuous_scale="Blues",
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    region = (
        filtered.groupby("region", as_index=False)
        .agg(total_sales=("sales", "sum"))
    )
    products = (
        filtered.groupby("product_name", as_index=False)
        .agg(total_sales=("sales", "sum"))
        .nlargest(10, "total_sales")
        .sort_values("total_sales")
    )

    c3, c4 = st.columns(2)
    with c3:
        fig = px.pie(
            region, names="region", values="total_sales",
            hole=0.55, title="Sales Contribution by Region",
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    with c4:
        fig = px.bar(
            products, x="total_sales", y="product_name",
            orientation="h", title="Top 10 Products by Sales",
            labels={"total_sales": "Sales", "product_name": "Product"},
            color="total_sales", color_continuous_scale="Purples",
        )
        st.plotly_chart(chart_layout(fig, 440), use_container_width=True)


# ============================================================
# PAGE 2: FACTORY ANALYSIS
# ============================================================

elif page == "Factory Analysis":
    show_page_header(
        "Factory Analysis",
        "Compare factory sales, order activity and gross profit contribution.",
    )

    factory = (
        filtered.groupby("factory", as_index=False)
        .agg(
            total_sales=("sales", "sum"),
            total_orders=("order_id", "nunique"),
            total_units=("units", "sum"),
            total_gross_profit=("gross_profit", "sum"),
        )
    )
    factory["gross_profit_margin_pct"] = np.where(
        factory["total_sales"] != 0,
        factory["total_gross_profit"] / factory["total_sales"] * 100,
        0,
    )
    factory["sales_share_pct"] = (
        factory["total_sales"] / factory["total_sales"].sum() * 100
    )

    show_kpis(filtered)
    st.divider()

    best_factory = factory.loc[factory["total_sales"].idxmax()]
    highest_margin_factory = factory.loc[
        factory["gross_profit_margin_pct"].idxmax()
    ]

    insight1, insight2 = st.columns(2)
    insight1.info(
        f"Highest sales: **{best_factory['factory']}** "
        f"({best_factory['total_sales']:,.2f})"
    )
    insight2.info(
        f"Highest gross profit margin: **{highest_margin_factory['factory']}** "
        f"({highest_margin_factory['gross_profit_margin_pct']:.2f}%)"
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            factory.sort_values("total_sales"),
            x="total_sales", y="factory", orientation="h",
            title="Factory Sales Ranking",
            labels={"total_sales": "Sales", "factory": "Factory"},
            color="total_sales", color_continuous_scale="Blues",
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    with c2:
        fig = px.bar(
            factory.sort_values("gross_profit_margin_pct"),
            x="gross_profit_margin_pct", y="factory", orientation="h",
            title="Gross Profit Margin by Factory",
            labels={
                "gross_profit_margin_pct": "Gross Profit Margin (%)",
                "factory": "Factory",
            },
            color="gross_profit_margin_pct",
            color_continuous_scale="Teal",
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    st.subheader("Factory Performance Details")
    st.dataframe(
        factory.sort_values("total_sales", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Factory Sales Contribution")
    contribution = factory.sort_values("sales_share_pct", ascending=False)
    fig = px.pie(
        contribution,
        names="factory",
        values="sales_share_pct",
        hole=0.55,
        title="Share of Filtered Sales by Factory",
    )
    st.plotly_chart(chart_layout(fig), use_container_width=True)


# ============================================================
# PAGE 3: PRODUCT ANALYSIS
# ============================================================

elif page == "Product Analysis":
    show_page_header(
        "Product Analysis",
        "Understand product sales, units sold and gross profit contribution.",
    )

    product = (
        filtered.groupby("product_name", as_index=False)
        .agg(
            total_sales=("sales", "sum"),
            total_units=("units", "sum"),
            total_gross_profit=("gross_profit", "sum"),
            total_orders=("order_id", "nunique"),
        )
    )
    product["gross_profit_margin_pct"] = np.where(
        product["total_sales"] != 0,
        product["total_gross_profit"] / product["total_sales"] * 100,
        0,
    )
    product["sales_share_pct"] = (
        product["total_sales"] / product["total_sales"].sum() * 100
    )

    show_kpis(filtered)
    st.divider()

    # ABC classification based on cumulative sales contribution.
    product = product.sort_values(
        "total_sales", ascending=False
    ).reset_index(drop=True)

    product["sales_contribution_pct"] = (
        product["total_sales"] / product["total_sales"].sum() * 100
    )
    product["cumulative_sales_pct"] = (
        product["sales_contribution_pct"].cumsum()
    )
    product["ABC_Category"] = np.select(
        [
            product["cumulative_sales_pct"] <= 80,
            product["cumulative_sales_pct"] <= 95,
        ],
        ["A", "B"],
        default="C",
    )

    abc_summary = (
        product.groupby("ABC_Category", as_index=False)
        .agg(
            product_count=("product_name", "count"),
            sales=("total_sales", "sum"),
            sales_contribution_pct=("sales_contribution_pct", "sum"),
        )
    )

    st.subheader("ABC Product Classification")
    st.caption(
        "A: cumulative sales up to 80%; B: above 80% up to 95%; "
        "C: remaining contribution. These are analytical thresholds."
    )
    st.dataframe(abc_summary, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        top_products = product.nlargest(10, "total_sales").sort_values("total_sales")
        fig = px.bar(
            top_products, x="total_sales", y="product_name",
            orientation="h", title="Top 10 Products by Sales",
            labels={"total_sales": "Sales", "product_name": "Product"},
            color="total_sales", color_continuous_scale="Purples",
        )
        st.plotly_chart(chart_layout(fig, 440), use_container_width=True)

    with c2:
        fig = px.scatter(
            product, x="total_units", y="total_sales",
            size="total_gross_profit", color="product_name",
            hover_name="product_name",
            title="Product Sales vs Units Sold",
            labels={"total_units": "Units Sold", "total_sales": "Sales"},
        )
        st.plotly_chart(chart_layout(fig, 440), use_container_width=True)

    st.subheader("Product Performance Details")
    st.dataframe(
        product.sort_values("total_sales", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE 4: GEOGRAPHY AND ROUTES
# ============================================================

elif page == "Geography & Routes":
    show_page_header(
        "Geography & Routes",
        "Explore regional sales, leading cities and factory-to-destination combinations.",
    )

    show_kpis(filtered)
    st.divider()

    region = (
        filtered.groupby("region", as_index=False)
        .agg(total_sales=("sales", "sum"), total_orders=("order_id", "nunique"))
    )
    city = (
        filtered.groupby(
            ["country_region", "state_province", "city"], as_index=False
        )
        .agg(total_sales=("sales", "sum"), total_orders=("order_id", "nunique"))
        .nlargest(15, "total_sales")
        .sort_values("total_sales")
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(
            region, names="region", values="total_sales",
            hole=0.55, title="Regional Sales Contribution",
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    with c2:
        fig = px.bar(
            city, x="total_sales", y="city", orientation="h",
            color="total_sales", title="Top 15 Cities by Sales",
            labels={"total_sales": "Sales", "city": "City"},
        )
        st.plotly_chart(chart_layout(fig, 440), use_container_width=True)

    st.subheader("State / Province Sales Concentration")
    state = (
        filtered.groupby(
            ["country_region", "state_province"], as_index=False
        )
        .agg(
            total_sales=("sales", "sum"),
            total_orders=("order_id", "nunique"),
        )
        .sort_values("total_sales", ascending=False)
    )
    state["sales_contribution_pct"] = (
        state["total_sales"] / state["total_sales"].sum() * 100
    )
    st.dataframe(state.head(15), use_container_width=True, hide_index=True)

    st.subheader("Factory vs Region Sales Matrix")
    factory_region_matrix = pd.pivot_table(
        filtered,
        index="factory",
        columns="region",
        values="sales",
        aggfunc="sum",
        fill_value=0,
    )
    fig = px.imshow(
        factory_region_matrix,
        text_auto=".0f",
        aspect="auto",
        title="Factory and Region Sales",
        labels={"x": "Region", "y": "Factory", "color": "Sales"},
    )
    st.plotly_chart(chart_layout(fig, 420), use_container_width=True)

    route = (
        filtered.groupby("route_id", as_index=False)
        .agg(
            factory=("factory", "first"),
            city=("city", "first"),
            state_province=("state_province", "first"),
            total_sales=("sales", "sum"),
            total_orders=("order_id", "nunique"),
            total_gross_profit=("gross_profit", "sum"),
        )
        .sort_values("total_sales", ascending=False)
    )

    st.subheader("Top Factory-to-Destination Combinations")
    st.caption(
        "Route means a factory and destination combination, not a measured road route."
    )
    st.dataframe(route.head(20), use_container_width=True, hide_index=True)


# ============================================================
# PAGE 5: SHIPPING ANALYSIS
# ============================================================

elif page == "Shipping Analysis":
    show_page_header(
        "Shipping Analysis",
        "Compare shipping-mode sales and order activity. This page does not infer delivery speed.",
    )

    shipping = (
        filtered.groupby("ship_mode", as_index=False)
        .agg(
            total_sales=("sales", "sum"),
            total_units=("units", "sum"),
            total_orders=("order_id", "nunique"),
            total_gross_profit=("gross_profit", "sum"),
        )
    )

    show_kpis(filtered)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            shipping, x="ship_mode", y="total_sales",
            color="ship_mode", title="Sales by Shipping Mode",
            labels={"ship_mode": "Shipping Mode", "total_sales": "Sales"},
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    with c2:
        fig = px.pie(
            shipping, names="ship_mode", values="total_sales",
            hole=0.55, title="Shipping Mode Sales Share",
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    st.subheader("Shipping Mode Details")
    st.dataframe(
        shipping.sort_values("total_sales", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Shipping Mode Sales by Order Year")
    shipping_year = (
        filtered.groupby(
            ["order_year", "ship_mode"], as_index=False
        )
        .agg(
            total_sales=("sales", "sum"),
            total_orders=("order_id", "nunique"),
        )
    )
    fig = px.bar(
        shipping_year,
        x="order_year",
        y="total_sales",
        color="ship_mode",
        barmode="group",
        title="Shipping Mode Sales by Order Year",
        labels={
            "order_year": "Order Year",
            "total_sales": "Sales",
            "ship_mode": "Shipping Mode",
        },
    )
    st.plotly_chart(chart_layout(fig, 420), use_container_width=True)

    st.subheader("Recorded Order and Ship Date Difference")
    date_check = filtered[
        ["order_date", "ship_date", "ship_mode"]
    ].dropna().copy()
    date_check["recorded_date_difference_days"] = (
        date_check["ship_date"] - date_check["order_date"]
    ).dt.days

    if not date_check.empty:
        st.dataframe(
            date_check["recorded_date_difference_days"].describe()
            .to_frame("Recorded Date Difference (Days)"),
            use_container_width=True,
        )

    st.warning(
        "The source data's order and ship dates require date-key investigation. "
        "Do not interpret this difference as actual delivery lead time or compare "
        "shipping speed until the date mapping is resolved."
    )


# ============================================================
# PAGE 6: TRANSACTION EXPLORER
# ============================================================

elif page == "Transaction Explorer":
    show_page_header(
        "Transaction Explorer",
        "Search and inspect the transaction records selected by the global filters.",
    )

    search_text = st.text_input(
        "Search Order ID, City, Product or Factory",
        placeholder="Enter a keyword...",
    )

    view = filtered.copy()

    if search_text.strip():
        term = search_text.strip().lower()
        searchable = (
            view["order_id"].astype(str).str.lower().str.contains(term, na=False)
            | view["city"].astype(str).str.lower().str.contains(term, na=False)
            | view["product_name"].astype(str).str.lower().str.contains(term, na=False)
            | view["factory"].astype(str).str.lower().str.contains(term, na=False)
        )
        view = view[searchable]

    st.caption(f"Records displayed: {len(view):,}")
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Nassau Candy Distributor | Python • Pandas • Streamlit • Plotly"
)
st.caption(
    "Sales and gross profit use the supplied dataset. "
    "No GPS distance, transportation cost or verified delivery-time measure is available."
)
