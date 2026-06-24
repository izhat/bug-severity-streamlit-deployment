
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Error Analysis",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

STAGE1_DIR = (
    PROJECT_ROOT
    / "models"
    / "stage1"
    / "aggressive"
)

STAGE1_ERROR_DIR = (
    STAGE1_DIR
    / "error_analysis_report"
)

STAGE2_DIR = (
    PROJECT_ROOT
    / "models"
    / "stage2"
    / "dashboard_outputs"
)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path, low_memory=False)


def find_text_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "summary",
        "text_final",
        "model_text",
        "bug_report_text",
        "description",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def find_probability_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column

    return None


def display_error_table(
    df: pd.DataFrame,
    title: str,
    probability_candidates: list[str],
    key_prefix: str,
) -> None:
    st.subheader(title)

    if df.empty:
        st.warning("No error records were found.")
        return

    st.caption(f"{len(df):,} misclassified reports")

    filtered = df.copy()

    text_column = find_text_column(filtered)
    probability_column = find_probability_column(
        filtered,
        probability_candidates,
    )

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        search_text = st.text_input(
            "Search report text",
            key=f"{key_prefix}_search",
            placeholder="Enter a keyword",
        )

    with filter_col2:
        if probability_column is not None:
            probability_values = pd.to_numeric(
                filtered[probability_column],
                errors="coerce",
            )

            minimum_probability = st.slider(
                f"Minimum {probability_column}",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                key=f"{key_prefix}_probability",
            )

            filtered = filtered[
                probability_values >= minimum_probability
            ]

    if search_text and text_column is not None:
        filtered = filtered[
            filtered[text_column]
            .fillna("")
            .astype(str)
            .str.contains(
                search_text,
                case=False,
                regex=False,
            )
        ]

    preferred_columns = [
        "bug_id",
        "id",
        "summary",
        "component",
        "type",
        "platform",
        "op_sys",
        "initial_priority",
        "initial_priority_clean",
        "severity_clean",
        "severity_lm",
        "true_severity_lm",
        "pred_severity_lm",
        "label_binary",
        "label_lm",
        "pred_lm",
        "prob_high",
        "prob_medium",
        "final_prob_high",
        "final_prob_medium",
        "threshold_used",
    ]

    display_columns = [
        column
        for column in preferred_columns
        if column in filtered.columns
    ]

    if not display_columns:
        display_columns = filtered.columns.tolist()

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
        height=450,
    )

    st.download_button(
        "Download filtered errors",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"{key_prefix}_filtered.csv",
        mime="text/csv",
        key=f"{key_prefix}_download",
    )


st.title("Error Analysis")
st.caption(
    "Inspection of cross-project misclassifications from "
    "Firefox → Thunderbird"
)

stage1_tab, stage2_tab = st.tabs(
    [
        "Stage-1 errors",
        "Stage-2 errors",
    ]
)


# ============================================================
# STAGE-1
# ============================================================

with stage1_tab:
    st.header("Stage-1: HIGH vs NOT_HIGH")

    false_high_df = load_csv(
        STAGE1_DIR / "false_positives.csv"
    )

    missed_high_df = load_csv(
        STAGE1_DIR / "missed_high.csv"
    )

    summary_df = load_csv(
        STAGE1_ERROR_DIR / "stage1_error_summary.csv"
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "False HIGH predictions",
        f"{len(false_high_df):,}",
    )

    metric2.metric(
        "Missed HIGH bugs",
        f"{len(missed_high_df):,}",
    )

    metric3.metric(
        "Total Stage-1 errors",
        f"{len(false_high_df) + len(missed_high_df):,}",
    )

    if not summary_df.empty:
        with st.expander("Stage-1 error summary"):
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
            )

    error_type = st.radio(
        "Select Stage-1 error type",
        options=[
            "Missed HIGH",
            "False HIGH",
        ],
        horizontal=True,
    )

    if error_type == "Missed HIGH":
        display_error_table(
            missed_high_df,
            "HIGH bugs predicted as NOT_HIGH",
            [
                "prob_high",
                "final_prob_high",
                "probability_high",
            ],
            "stage1_missed_high",
        )
    else:
        display_error_table(
            false_high_df,
            "NOT_HIGH bugs predicted as HIGH",
            [
                "prob_high",
                "final_prob_high",
                "probability_high",
            ],
            "stage1_false_high",
        )


# ============================================================
# STAGE-2
# ============================================================

with stage2_tab:
    st.header("Stage-2: LOW vs MEDIUM")

    medium_as_low_df = load_csv(
        STAGE2_DIR
        / "errors_medium_predicted_low.csv"
    )

    low_as_medium_df = load_csv(
        STAGE2_DIR
        / "errors_low_predicted_medium.csv"
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "MEDIUM predicted LOW",
        f"{len(medium_as_low_df):,}",
    )

    metric2.metric(
        "LOW predicted MEDIUM",
        f"{len(low_as_medium_df):,}",
    )

    metric3.metric(
        "Total Stage-2 errors",
        f"{len(medium_as_low_df) + len(low_as_medium_df):,}",
    )

    error_type = st.radio(
        "Select Stage-2 error type",
        options=[
            "MEDIUM predicted LOW",
            "LOW predicted MEDIUM",
        ],
        horizontal=True,
    )

    if error_type == "MEDIUM predicted LOW":
        display_error_table(
            medium_as_low_df,
            "MEDIUM bugs incorrectly predicted as LOW",
            [
                "prob_medium",
                "final_prob_medium",
            ],
            "stage2_medium_as_low",
        )
    else:
        display_error_table(
            low_as_medium_df,
            "LOW bugs incorrectly predicted as MEDIUM",
            [
                "prob_medium",
                "final_prob_medium",
            ],
            "stage2_low_as_medium",
        )


st.info(
    "These error tables support qualitative analysis of "
    "boundary cases, misleading lexical signals, missing metadata, "
    "and cross-project vocabulary differences."
)

