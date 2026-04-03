import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import pickle

st.set_page_config(
    page_title="Instacart Recommender",
    page_icon="🛒",
    layout="wide",
)

# local is 8000, in docker the env var takes precedence
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# hit rates measured on 10,000 users
MODEL_INFO = {
    "ALS":  {"hit_rate": "16.2%", "description": "Alternating Least Squares"},
    "EASE": {"hit_rate": "14.6%", "description": "Embarrassingly Shallow Autoencoder"},
    "BPR":  {"hit_rate": "14.6%", "description": "Bayesian Personalized Ranking"},
}

@st.cache_data
def load_data():
    # nrows=5M to not blow up RAM (full file is ~32M rows)
    df = pd.read_csv(
        "data/processed/prior_merged.csv",
        usecols=["user_id", "order_id", "product_id", "product_name", "department", "reordered"],
        nrows=5_000_000
    )
    with open("data/processed/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    return df, mappings

df, mappings = load_data()

# product → department table, useful for cards
product_dept = df[["product_name", "department"]].drop_duplicates("product_name")
dept_lookup  = dict(zip(product_dept["product_name"], product_dept["department"]))

st.title("🛒 Instacart Recommendation System")
st.markdown(
    "Select a user to see their purchase history "
    "and the products the model recommends for them."
)
st.divider()

with st.sidebar:
    st.header("Parameters")
    user_id    = st.number_input("User ID", min_value=1, max_value=206209, value=1, step=1)
    n_recs     = st.slider("Number of recommendations", 5, 20, 10)
    model_name = st.selectbox("Model", ["ALS", "EASE", "BPR"])
    st.divider()
    st.markdown("**Selected model**")
    st.caption(MODEL_INFO[model_name]["description"])
    st.caption(f"Hit Rate @10 = **{MODEL_INFO[model_name]['hit_rate']}**")
    st.caption("206k users · 50k products")
    st.divider()
    st.caption("API: " + API_URL)

col1, col2 = st.columns(2)
recs = None  # initialized here to avoid errors at the bottom of the page

with col1:
    st.subheader("Purchase History")

    user_data    = df[df["user_id"] == user_id]
    user_history = (
        user_data.groupby("product_name")["order_id"]
        .count()
        .reset_index(name="nb_purchases")
        .sort_values("nb_purchases", ascending=False)
        .head(15)
    )

    if user_history.empty:
        st.warning("User not found.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Orders",            user_data["order_id"].nunique())
        m2.metric("Distinct Products", user_data["product_id"].nunique())
        m3.metric("Reorder Rate",      f"{user_data['reordered'].mean():.0%}")

        dept_counts = (
            user_data.groupby("department")
            .size()
            .reset_index(name="purchases")
            .sort_values("purchases", ascending=False)
            .head(6)
        )
        fig_dept = px.bar(
            dept_counts, x="purchases", y="department",
            orientation="h", color="purchases",
            color_continuous_scale=["#c9dff2", "#5A8ABB"],
            title="Preferred Departments",
        )
        fig_dept.update_layout(
            height=280, showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dept, use_container_width=True)

        st.dataframe(
            user_history.rename(columns={"product_name": "Product", "nb_purchases": "Purchases"}),
            use_container_width=True, hide_index=True,
        )

with col2:
    st.subheader(f"Recommendations — {model_name}")

    try:
        with st.spinner("Loading recommendations..."):
            response = requests.get(
                f"{API_URL}/recommend/{user_id}?n={n_recs}&model_name={model_name.lower()}"
            )
            response.raise_for_status()
            recs = pd.DataFrame(response.json()["recommendations"])

        fig_recs = px.bar(
            recs.sort_values("score"),
            x="score", y="product",
            orientation="h", color="score",
            color_continuous_scale=["#c9e8d0", "#2e8b57"],
            title=f"Top {n_recs} recommendations — user {user_id}",
        )
        fig_recs.update_layout(
            height=380, showlegend=False, coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_recs, use_container_width=True)

        st.markdown("**Recommended Products**")
        recs_display = recs.copy()
        recs_display["Department"] = recs_display["product"].map(
            lambda p: dept_lookup.get(p, "—")
        )
        recs_display["score"] = recs_display["score"].round(4)
        recs_display = recs_display.rename(columns={
            "product": "Product",
            "score":   "Score",
        })
        st.dataframe(
            recs_display[["Product", "Score", "Department"]],
            use_container_width=True,
            hide_index=True,
        )

    except requests.exceptions.ConnectionError:
        st.error("API not available — launch uvicorn first.")
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()
st.subheader("Consistency — recommended vs purchased departments")

if recs is not None and not user_history.empty:
    rec_depts = (
        product_dept[product_dept["product_name"].isin(recs["product"].tolist())]
        ["department"].value_counts().reset_index()
    )
    rec_depts.columns = ["department", "count"]
    rec_depts["source"] = "Recommended"

    hist_depts = dept_counts.rename(columns={"purchases": "count"}).copy()
    hist_depts["source"] = "History"

    fig_compare = px.bar(
        pd.concat([hist_depts, rec_depts]),
        x="department", y="count", color="source",
        barmode="group",
        color_discrete_map={"History": "#5A8ABB", "Recommended": "#C9A84C"},
        title="Department comparison: history vs recommendations",
    )
    fig_compare.update_layout(
        height=350, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_compare, use_container_width=True)