import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH = "data/processed/osha_clean.csv"

AGENT_DIR = Path(__file__).resolve().parent.parent / "src" / "agent"
sys.path.insert(0, str(AGENT_DIR))

import graph  # noqa: E402 -- provides the compiled GRAPH; do not reimplement its logic here
import test_retrieval  # noqa: E402
import tools  # noqa: E402

st.set_page_config(page_title="OSHA Construction Fatality Risk Dashboard", layout="wide")


# tools.py/test_retrieval.py already lazy-singleton these behind plain module
# globals (no Streamlit dependency there, since they're also used by the
# standalone CLI scripts) -- these wrappers just make that caching explicit
# and Streamlit-tracked at the app layer, and guarantee the heavy loads
# (torch/sentence-transformers, Chroma, the two joblib models) happen once
# per server process rather than once per chat message.
@st.cache_resource(show_spinner="Loading embedding model and Chroma index...")
def get_retrieval_resources():
    return test_retrieval._get_model_and_collection()


@st.cache_resource(show_spinner="Loading Day 4/5 risk models...")
def get_risk_models():
    return tools._load_day4(), tools._load_day5()


@st.cache_resource
def get_agent_graph():
    return graph.GRAPH


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
    return df


df = load_data()

# ---- 1. Header ----
st.title("OSHA Construction Fatality Risk Dashboard")
st.warning(
    "**This dataset's ~79% fatality rate is not a real construction-industry "
    "statistic.** It reflects which incidents the source site chose to scrape "
    "full narratives for, not a representative sample of all OSHA construction "
    "incidents. See `docs/model_card.md` for full detail.",
    icon="⚠️",
)

tab_dashboard, tab_chat = st.tabs(["Dashboard", "Risk & Guidance Chat"])

with tab_dashboard:
    # ---- Sidebar filters (apply to the dashboard tab) ----
    st.sidebar.header("Filters")

    occupation_options = sorted(df["occupation_primary"].dropna().unique())
    selected_occupations = st.sidebar.multiselect("Occupation", occupation_options)

    industry_options = sorted(df["industry_name"].dropna().unique())
    selected_industries = st.sidebar.multiselect("Industry", industry_options)

    fatality_choice = st.sidebar.radio("Fatality status", ["All", "Fatal", "Non-fatal"])

    filtered_df = df.copy()
    if selected_occupations:
        filtered_df = filtered_df[filtered_df["occupation_primary"].isin(selected_occupations)]
    if selected_industries:
        filtered_df = filtered_df[filtered_df["industry_name"].isin(selected_industries)]
    if fatality_choice == "Fatal":
        filtered_df = filtered_df[filtered_df["is_fatality"]]
    elif fatality_choice == "Non-fatal":
        filtered_df = filtered_df[~filtered_df["is_fatality"]]

    st.sidebar.caption(f"{len(filtered_df):,} of {len(df):,} incidents match the current filters.")

    # ---- 2. Key metrics ----
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total incidents", f"{len(filtered_df):,}")

    fatality_rate = filtered_df["is_fatality"].mean() if len(filtered_df) else float("nan")
    col2.metric("Fatality rate", f"{fatality_rate * 100:.1f}%" if pd.notna(fatality_rate) else "N/A")

    most_common_injury = (
        filtered_df["injury_type"].mode().iat[0] if not filtered_df["injury_type"].dropna().empty else "N/A"
    )
    col3.metric("Most common injury type", most_common_injury)

    most_common_occupation = (
        filtered_df["occupation_primary"].mode().iat[0]
        if not filtered_df["occupation_primary"].dropna().empty
        else "N/A"
    )
    col4.metric("Most common occupation", most_common_occupation)

    # ---- 3. Industry volume + fatality rate, and monthly trend ----
    st.subheader("Incidents by Industry")
    industry_summary = (
        filtered_df.groupby("industry_name")
        .agg(incidents=("id", "count"), fatality_rate=("is_fatality", "mean"))
        .query("incidents >= 10")
        .reset_index()
        .sort_values("incidents", ascending=False)
    )
    if industry_summary.empty:
        st.info("No industry has at least 10 incidents under the current filters.")
    else:
        fig_industry = px.bar(
            industry_summary,
            x="industry_name",
            y="incidents",
            color="fatality_rate",
            color_continuous_scale="YlOrRd",
            title="Incidents by Industry (color = fatality rate, industries with >= 10 incidents)",
            labels={"industry_name": "Industry", "incidents": "Number of incidents", "fatality_rate": "Fatality rate"},
        )
        fig_industry.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_industry, width="stretch")

    st.subheader("Incidents Over Time")
    monthly_trend = (
        filtered_df.dropna(subset=["year", "month"])
        .assign(year=lambda d: d["year"].astype(int), month=lambda d: d["month"].astype(int))
        .groupby(["year", "month"])
        .size()
        .reset_index(name="incidents")
    )
    if monthly_trend.empty:
        st.info("No dated incidents under the current filters.")
    else:
        monthly_trend["date"] = pd.to_datetime(dict(year=monthly_trend["year"], month=monthly_trend["month"], day=1))
        fig_trend = px.line(
            monthly_trend,
            x="date",
            y="incidents",
            title="Incidents Over Time (monthly)",
            labels={"date": "Month", "incidents": "Number of incidents"},
        )
        fig_trend.update_traces(mode="lines+markers")
        st.plotly_chart(fig_trend, width="stretch")

    # ---- 5. Occupation x injury type heatmap ----
    st.subheader("Risk Heatmap: Occupation x Injury Type")
    top_occs = filtered_df["occupation_primary"].value_counts().head(12).index
    top_injuries = filtered_df["injury_type"].value_counts().head(12).index
    heat_df = filtered_df[
        filtered_df["occupation_primary"].isin(top_occs) & filtered_df["injury_type"].isin(top_injuries)
    ]
    if heat_df.empty:
        st.info("Not enough data to build a heatmap under the current filters.")
    else:
        pivot = pd.crosstab(heat_df["occupation_primary"], heat_df["injury_type"]).loc[top_occs, top_injuries]
        fig_heatmap = px.imshow(
            pivot,
            text_auto=True,
            color_continuous_scale="YlOrRd",
            aspect="auto",
            labels=dict(x="Injury type", y="Occupation", color="Incident count"),
            title="Incident Count: Top Occupations x Top Injury Types",
        )
        fig_heatmap.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_heatmap, width="stretch")

    # ---- 4. Filterable data table ----
    st.subheader("Incident Data")
    display_cols = [
        "id", "industry_name", "occupation_primary", "injury_type",
        "is_fatality", "is_fall_incident", "fall_distance_ft", "incident_date", "title",
    ]
    st.dataframe(filtered_df[display_cols], width="stretch", hide_index=True)

with tab_chat:
    st.caption(
        "Runs the same LangGraph pipeline as `src/agent/graph.py`: predict_risk -> "
        "retrieve_guidance -> draft_recommendation. Drafts are for a human safety "
        "officer to review, not to act on automatically."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["description"])
        with st.chat_message("assistant"):
            risk = entry.get("risk")
            if risk:
                rcol1, rcol2, rcol3 = st.columns(3)
                rcol1.metric("Fatality risk", f"{risk['probability']:.1%}")
                rcol2.metric("Confidence", "Low" if risk.get("low_confidence") else "Normal")
                rcol3.caption(risk.get("model_used", ""))
                if risk.get("caveat"):
                    st.info(risk["caveat"])

            if entry.get("guidance_text"):
                with st.expander(f"Retrieved guidance (query: \"{entry.get('guidance_query', '')}\")"):
                    st.text(entry["guidance_text"])

            if entry.get("draft"):
                st.markdown(entry["draft"])

            if entry.get("error"):
                st.error(entry["error"], icon="🚫")

    description = st.chat_input("Describe an incident...")
    if description:
        entry = {"description": description}
        with st.spinner("Predicting risk, retrieving guidance, and drafting a recommendation..."):
            try:
                get_retrieval_resources()
                get_risk_models()
                compiled_graph = get_agent_graph()
                for step_output in compiled_graph.stream({"description": description}, stream_mode="updates"):
                    for _node_name, node_state in step_output.items():
                        entry.update(node_state)
            except Exception as e:
                # Most commonly a missing GOOGLE_API_KEY at the draft_recommendation
                # step -- graph.py already raises a clear, specific message for that
                # case, so surface it as-is rather than a generic failure or a crash.
                entry["error"] = str(e)

        st.session_state.chat_history.append(entry)
        st.rerun()
