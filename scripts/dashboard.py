import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

DATA_PATH   = "data/processed/prior_merged.csv"
OUTPUT_PATH = "data/processed/plots/dashboard.html"
os.makedirs("data/processed/plots", exist_ok=True)

df = pd.read_csv(
    DATA_PATH,
    usecols=["user_id", "order_id", "product_id", "product_name", "aisle",
             "department", "reordered", "order_dow", "order_hour_of_day"],
    dtype={
        "user_id": "int32",
        "order_id": "int32",
        "product_id": "int32",
        "reordered": "int8",
        "order_dow": "int8",
        "order_hour_of_day": "int8",
    },
    nrows=5_000_000,
)
reordered = df[df["reordered"] == 1]

# global metrics
n_users       = df["user_id"].nunique()
n_products    = df["product_id"].nunique()
n_orders      = df["order_id"].nunique()
reorder_rate  = df["reordered"].mean() * 100

# data for charts
top_products = reordered["product_name"].value_counts().head(20).reset_index()
top_products.columns = ["product_name", "count"]

top_aisles = reordered["aisle"].value_counts().head(20).reset_index()
top_aisles.columns = ["aisle", "count"]

dept_rate = df.groupby("department")["reordered"].mean().reset_index()
dept_rate.columns = ["department", "rate"]
dept_rate["rate"] = (dept_rate["rate"] * 100).round(2)
dept_rate = dept_rate.sort_values("rate", ascending=False)

# heatmap department × day
days = {0: "Saturday", 1: "Sunday", 2: "Monday", 3: "Tuesday",
        4: "Wednesday", 5: "Thursday", 6: "Friday"}
df["day"] = df["order_dow"].map(days)
heatmap_pivot = (
    df.groupby(["department", "day"])["reordered"].mean() * 100
).reset_index().pivot(index="department", columns="day", values="reordered")[list(days.values())]

orders_per_user        = df.groupby("user_id")["order_id"].nunique()
products_per_order     = df.groupby("order_id")["product_id"].nunique()

purchases_per_hour = df["order_hour_of_day"].value_counts().sort_index().reset_index()
purchases_per_hour.columns = ["hour", "count"]

purchases_per_day = df["day"].value_counts().reindex(list(days.values())).reset_index()
purchases_per_day.columns = ["day", "count"]

print("Building dashboard...")
figures = []

# KPI cards
fig_kpi = go.Figure()
kpis = [
    ("Users", f"{n_users:,}"),
    ("Products", f"{n_products:,}"),
    ("Orders", f"{n_orders:,}"),
    ("Reorder Rate", f"{reorder_rate:.1f}%"),
]
for i, (label, value) in enumerate(kpis):
    fig_kpi.add_trace(go.Indicator(
        mode="number",
        value=None,
        title={"text": f"<b>{value}</b><br><span style='font-size:14px'>{label}</span>"},
        domain={"x": [i * 0.25, (i + 1) * 0.25], "y": [0, 1]}
    ))
fig_kpi.update_layout(title="Dataset Overview", height=150)
figures.append(fig_kpi)

fig_products = px.bar(
    top_products.sort_values("count"), x="count", y="product_name",
    orientation="h", title="Top 20 Most Reordered Products",
    labels={"count": "Number of Reorders", "product_name": "Product"},
    color="count", color_continuous_scale="Blues"
)
fig_products.update_layout(height=600, showlegend=False)
figures.append(fig_products)

fig_aisles = px.bar(
    top_aisles.sort_values("count"), x="count", y="aisle",
    orientation="h", title="Top 20 Most Reordered Aisles",
    labels={"count": "Number of Reorders", "aisle": "Aisle"},
    color="count", color_continuous_scale="Teal"
)
fig_aisles.update_layout(height=600, showlegend=False)
figures.append(fig_aisles)

fig_rate = px.bar(
    dept_rate.sort_values("rate"), x="rate", y="department",
    orientation="h", title="Reorder Rate by Department (%)",
    labels={"rate": "Reorder Rate (%)", "department": "Department"},
    color="rate", color_continuous_scale="RdYlGn"
)
fig_rate.update_layout(height=500, showlegend=False)
figures.append(fig_rate)

fig_heatmap = px.imshow(
    heatmap_pivot,
    title="Reorder Rate (%) by Department and Day of Week",
    labels={"x": "Day", "y": "Department", "color": "Rate (%)"},
    color_continuous_scale="YlOrRd", aspect="auto"
)
fig_heatmap.update_layout(height=600)
figures.append(fig_heatmap)

fig_dist_user = px.histogram(
    orders_per_user, nbins=50,
    title="Distribution of Orders per User",
    labels={"value": "Number of Orders", "count": "Number of Users"}
)
fig_dist_user.update_layout(height=400, showlegend=False)
figures.append(fig_dist_user)

fig_dist_order = px.histogram(
    products_per_order, nbins=50,
    title="Distribution of Products per Order",
    labels={"value": "Number of Products", "count": "Number of Orders"}
)
fig_dist_order.update_layout(height=400, showlegend=False)
figures.append(fig_dist_order)

fig_hour = px.line(
    purchases_per_hour, x="hour", y="count",
    title="Purchases by Hour of Day",
    labels={"hour": "Hour", "count": "Number of Purchases"}, markers=True
)
fig_hour.update_layout(height=400)
figures.append(fig_hour)

fig_day = px.bar(
    purchases_per_day, x="day", y="count",
    title="Purchases by Day of Week",
    labels={"day": "Day", "count": "Number of Purchases"},
    color="count", color_continuous_scale="Blues"
)
fig_day.update_layout(height=400, showlegend=False)
figures.append(fig_day)

# HTML export
print("Exporting dashboard...")
html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Instacart EDA Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }
        h1 { text-align: center; color: #2c3e50; }
        .plot-container { background: white; border-radius: 8px;
                          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                          margin-bottom: 24px; padding: 16px; }
    </style>
</head>
<body>
<h1>Instacart: Exploratory Data Analysis</h1>
"""

for fig in figures:
    html_content += f'<div class="plot-container">{fig.to_html(full_html=False, include_plotlyjs="cdn")}</div>\n'

html_content += "</body></html>"

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Dashboard saved → {OUTPUT_PATH}")