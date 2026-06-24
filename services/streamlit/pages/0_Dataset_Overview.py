
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Dataset Overview",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_ROOT
    / "services"
    / "streamlit"
    / "dashboard_data"
)


@st.cache_data(show_spinner="Loading dashboard dataset summary...")
def load_dashboard_assets() -> dict:
    with open(
        DATA_DIR / "dataset_summary.json",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    return {
        "summary": summary,
        "project_counts": pd.read_csv(
            DATA_DIR / "project_counts.csv"
        ),
        "severity_counts": pd.read_csv(
            DATA_DIR / "severity_counts.csv"
        ),
        "project_severity_counts": pd.read_csv(
            DATA_DIR / "project_severity_counts.csv"
        ),
        "stage1_counts": pd.read_csv(
            DATA_DIR / "stage1_counts.csv"
        ),
        "metadata_completeness": pd.read_csv(
            DATA_DIR / "metadata_completeness.csv"
        ),
        "sample": pd.read_csv(
            DATA_DIR / "dataset_sample.csv",
            low_memory=False,
        ),
    }


required_files = [
    "dataset_summary.json",
    "project_counts.csv",
    "severity_counts.csv",
    "project_severity_counts.csv",
    "stage1_counts.csv",
    "metadata_completeness.csv",
    "dataset_sample.csv",
]

missing_files = [
    filename
    for filename in required_files
    if not (DATA_DIR / filename).exists()
]

if missing_files:
    st.error(
        "Missing dashboard dataset assets: "
        + ", ".join(missing_files)
    )
    st.stop()


assets = load_dashboard_assets()

summary = assets["summary"]
project_counts = assets["project_counts"]
severity_counts = assets["severity_counts"]
project_severity = assets["project_severity_counts"]
stage1_counts = assets["stage1_counts"]
metadata_completeness = assets["metadata_completeness"]
sample_df = assets["sample"]


st.title("Dataset Overview")
st.caption(
    "Final cleaned and enriched BugsRepo dataset used for "
    "Firefox → Thunderbird cross-project severity prediction"
)


# ============================================================
# SUMMARY
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Final dataset rows",
    f"{summary['total_rows']:,}",
)

metric2.metric(
    "Dataset columns",
    f"{summary['total_columns']:,}",
)

metric3.metric(
    "Projects",
    f"{summary['project_count']:,}",
)

metric4.metric(
    "HIGH-severity rows",
    f"{summary['high_count']:,}",
)


# ============================================================
# CLEANING
# ============================================================

st.subheader("Final cleaning stage")

cleaning1, cleaning2, cleaning3 = st.columns(3)

cleaning1.metric(
    "Rows before final cleaning",
    f"{summary['rows_before_final_cleaning']:,}",
)

cleaning2.metric(
    "Rows retained",
    f"{summary['rows_retained']:,}",
)

removed_percentage = (
    summary["rows_removed"]
    / summary["rows_before_final_cleaning"]
)

cleaning3.metric(
    "Rows removed",
    f"{summary['rows_removed']:,}",
    delta=f"-{removed_percentage:.2%}",
)

cleaning_df = pd.DataFrame(
    {
        "Removal reason": [
            "Missing usable bug text",
            "CI/log-like reports",
        ],
        "Rows removed": [
            summary["missing_text_removed"],
            summary["ci_log_like_removed"],
        ],
    }
)

with st.expander("Final cleaning breakdown"):
    st.dataframe(
        cleaning_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PROJECT DISTRIBUTION
# ============================================================

st.subheader("Project distribution")

maximum_projects = min(20, len(project_counts))

project_limit = st.slider(
    "Number of projects to display",
    min_value=min(5, maximum_projects),
    max_value=maximum_projects,
    value=min(12, maximum_projects),
)

top_projects = project_counts.head(project_limit)

project_chart = (
    alt.Chart(top_projects)
    .mark_bar()
    .encode(
        x=alt.X(
            "Bug reports:Q",
            title="Number of bug reports",
        ),
        y=alt.Y(
            "Project:N",
            sort="-x",
            title="Project",
        ),
        tooltip=[
            "Project:N",
            alt.Tooltip(
                "Bug reports:Q",
                format=",",
            ),
        ],
    )
    .properties(
        height=max(300, project_limit * 32),
    )
)

st.altair_chart(
    project_chart,
    use_container_width=True,
)


# ============================================================
# THREE-CLASS DISTRIBUTION
# ============================================================

st.subheader("Three-class severity distribution")

severity_counts = severity_counts.copy()
severity_total = severity_counts["Bug reports"].sum()

severity_counts["Percentage"] = (
    severity_counts["Bug reports"]
    / severity_total
    * 100
)

severity_col1, severity_col2 = st.columns(2)

with severity_col1:
    severity_chart = (
        alt.Chart(severity_counts)
        .mark_bar()
        .encode(
            x=alt.X(
                "Severity:N",
                sort=["HIGH", "MEDIUM", "LOW"],
            ),
            y=alt.Y(
                "Bug reports:Q",
                title="Number of bug reports",
            ),
            tooltip=[
                "Severity:N",
                alt.Tooltip(
                    "Bug reports:Q",
                    format=",",
                ),
                alt.Tooltip(
                    "Percentage:Q",
                    format=".2f",
                ),
            ],
        )
        .properties(height=360)
    )

    st.altair_chart(
        severity_chart,
        use_container_width=True,
    )

with severity_col2:
    st.dataframe(
        severity_counts,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# STAGE-1 DISTRIBUTION
# ============================================================

st.subheader("Stage-1 HIGH vs NOT_HIGH distribution")

stage1_counts = stage1_counts.copy()
stage1_total = stage1_counts["Bug reports"].sum()

stage1_counts["Percentage"] = (
    stage1_counts["Bug reports"]
    / stage1_total
    * 100
)

binary1, binary2, binary3 = st.columns(3)

binary1.metric(
    "HIGH",
    f"{summary['high_count']:,}",
)

binary2.metric(
    "NOT_HIGH",
    f"{summary['not_high_count']:,}",
)

binary3.metric(
    "HIGH class share",
    f"{summary['high_count'] / summary['total_rows']:.2%}",
)

stage1_chart = (
    alt.Chart(stage1_counts)
    .mark_bar()
    .encode(
        x=alt.X(
            "Stage-1 class:N",
            sort=["HIGH", "NOT_HIGH"],
        ),
        y=alt.Y(
            "Bug reports:Q",
            title="Number of bug reports",
        ),
        tooltip=[
            "Stage-1 class:N",
            alt.Tooltip(
                "Bug reports:Q",
                format=",",
            ),
            alt.Tooltip(
                "Percentage:Q",
                format=".2f",
            ),
        ],
    )
    .properties(height=340)
)

st.altair_chart(
    stage1_chart,
    use_container_width=True,
)


# ============================================================
# PROJECT × SEVERITY
# ============================================================

st.subheader("Severity distribution by project")

project_options = project_counts["Project"].tolist()

selected_projects = st.multiselect(
    "Select projects",
    options=project_options,
    default=project_options[:6],
)

filtered_project_severity = project_severity[
    project_severity["project_display"].isin(
        selected_projects
    )
]

if filtered_project_severity.empty:
    st.info("Select at least one project.")
else:
    project_severity_chart = (
        alt.Chart(filtered_project_severity)
        .mark_bar()
        .encode(
            x=alt.X(
                "project_display:N",
                title="Project",
            ),
            y=alt.Y(
                "Bug reports:Q",
                title="Number of bug reports",
                stack="zero",
            ),
            color=alt.Color(
                "severity_three_class:N",
                title="Severity",
                sort=["HIGH", "MEDIUM", "LOW"],
            ),
            tooltip=[
                alt.Tooltip(
                    "project_display:N",
                    title="Project",
                ),
                alt.Tooltip(
                    "severity_three_class:N",
                    title="Severity",
                ),
                alt.Tooltip(
                    "Bug reports:Q",
                    format=",",
                ),
            ],
        )
        .properties(height=430)
    )

    st.altair_chart(
        project_severity_chart,
        use_container_width=True,
    )


# ============================================================
# METADATA COMPLETENESS
# ============================================================

st.subheader("Metadata completeness")

st.dataframe(
    metadata_completeness,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# SAMPLE EXPLORER
# ============================================================

st.subheader("Dataset sample explorer")

st.caption(
    "This deployment uses a reproducible 3,000-row sample for "
    "interactive exploration. All aggregate charts are calculated "
    "from the complete 52,099-row dataset."
)

sample_project_options = sorted(
    sample_df["project_display"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    filter_project = st.selectbox(
        "Filter by project",
        options=["All"] + sample_project_options,
    )

with filter_col2:
    filter_severity = st.multiselect(
        "Filter by severity",
        options=["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
    )

filtered_sample = sample_df.copy()

if filter_project != "All":
    filtered_sample = filtered_sample[
        filtered_sample["project_display"]
        == filter_project
    ]

filtered_sample = filtered_sample[
    filtered_sample["severity_three_class"].isin(
        filter_severity
    )
]

st.caption(
    f"{len(filtered_sample):,} sampled rows match the filters"
)

st.dataframe(
    filtered_sample.head(500),
    use_container_width=True,
    hide_index=True,
    height=500,
)

