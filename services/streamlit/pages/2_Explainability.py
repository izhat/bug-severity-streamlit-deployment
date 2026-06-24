
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Explainability",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

STAGE1_XAI_DIR = (
    PROJECT_ROOT
    / "models"
    / "stage1"
    / "shap_xai"
)

STAGE2_XAI_DIR = (
    PROJECT_ROOT
    / "models"
    / "stage2"
    / "dashboard_outputs"
)


def show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(
            str(path),
            caption=caption,
            use_container_width=True,
        )
    else:
        st.warning(f"Missing image: {path.name}")


def show_csv(path: Path, title: str) -> None:
    st.markdown(f"#### {title}")

    if path.exists():
        dataframe = pd.read_csv(
            path,
            low_memory=False,
        )

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning(f"Missing file: {path.name}")


st.title("Model Explainability")
st.caption(
    "SHAP-based interpretation of the Stage-1 and Stage-2 "
    "cross-project severity models"
)

stage1_tab, stage2_tab = st.tabs(
    [
        "Stage-1 explainability",
        "Stage-2 explainability",
    ]
)


# ============================================================
# STAGE-1 EXPLAINABILITY
# ============================================================

with stage1_tab:
    st.header("Stage-1: HIGH vs NOT_HIGH")

    st.write(
        "Stage-1 SHAP analysis explains which textual, "
        "GraphLens, process, and initial-priority features "
        "influence HIGH-severity predictions."
    )

    global_options = {
        "Global feature importance": (
            STAGE1_XAI_DIR
            / "shap_global_bar_top30.png"
        ),
        "SHAP summary plot": (
            STAGE1_XAI_DIR
            / "shap_summary_top25.png"
        ),
        "Feature-group importance": (
            STAGE1_XAI_DIR
            / "shap_group_importance.png"
        ),
        "GraphLens signal importance": (
            STAGE1_XAI_DIR
            / "shap_graphlens_signal_importance.png"
        ),
        "Initial-priority importance": (
            STAGE1_XAI_DIR
            / "shap_initial_priority_importance.png"
        ),
        "Metadata feature importance": (
            STAGE1_XAI_DIR
            / "shap_top_meta_features.png"
        ),
    }

    selected_global_plot = st.selectbox(
        "Select a Stage-1 global explanation",
        options=list(global_options.keys()),
        key="stage1_global_plot",
    )

    show_image(
        global_options[selected_global_plot],
        selected_global_plot,
    )

    st.divider()

    st.subheader("Local prediction explanations")

    local_options = {
        "Correct HIGH prediction": (
            STAGE1_XAI_DIR
            / "local_waterfall_highest_prob_high.png"
        ),
        "False HIGH prediction": (
            STAGE1_XAI_DIR
            / "local_waterfall_false_high_highest_prob.png"
        ),
        "Missed HIGH near decision boundary": (
            STAGE1_XAI_DIR
            / "local_waterfall_missed_high_near_boundary.png"
        ),
    }

    selected_local_plot = st.selectbox(
        "Select a local explanation case",
        options=list(local_options.keys()),
        key="stage1_local_plot",
    )

    show_image(
        local_options[selected_local_plot],
        selected_local_plot,
    )

    with st.expander("Local explanation case details"):
        show_csv(
            STAGE1_XAI_DIR
            / "local_explanation_cases.csv",
            "Selected Stage-1 explanation cases",
        )

    with st.expander("Stage-1 SHAP data tables"):
        table_options = {
            "All feature importance": (
                STAGE1_XAI_DIR
                / "shap_all_feature_importance.csv"
            ),
            "Feature-group importance": (
                STAGE1_XAI_DIR
                / "shap_group_importance.csv"
            ),
            "GraphLens importance": (
                STAGE1_XAI_DIR
                / "shap_graphlens_signal_importance.csv"
            ),
            "Initial-priority importance": (
                STAGE1_XAI_DIR
                / "shap_initial_priority_importance.csv"
            ),
            "Metadata importance": (
                STAGE1_XAI_DIR
                / "shap_meta_feature_importance.csv"
            ),
        }

        selected_table = st.selectbox(
            "Select Stage-1 SHAP table",
            options=list(table_options.keys()),
            key="stage1_shap_table",
        )

        show_csv(
            table_options[selected_table],
            selected_table,
        )


# ============================================================
# STAGE-2 EXPLAINABILITY
# ============================================================

with stage2_tab:
    st.header("Stage-2: LOW vs MEDIUM")

    st.write(
        "Stage-2 explainability shows how XGBoost, LightGBM, "
        "and LSTM probabilities contribute to the final stacked "
        "LOW versus MEDIUM prediction."
    )

    col1, col2 = st.columns(2)

    with col1:
        show_image(
            STAGE2_XAI_DIR
            / "shap_ensemble_model_contribution.png",
            "Global ensemble-model contribution",
        )

    with col2:
        show_image(
            STAGE2_XAI_DIR
            / "shap_ensemble_summary.png",
            "Stage-2 ensemble SHAP summary",
        )

    st.divider()

    st.subheader("Stage-2 contribution tables")

    table1, table2 = st.columns(2)

    with table1:
        show_csv(
            STAGE2_XAI_DIR
            / "shap_ensemble_model_contribution.csv",
            "Mean SHAP contribution by model",
        )

    with table2:
        show_csv(
            STAGE2_XAI_DIR
            / "stacked_ensemble_model_coefficients.csv",
            "Stacked calibrator coefficients",
        )

    with st.expander("Average model probability comparison"):
        show_image(
            STAGE2_XAI_DIR
            / "average_model_probabilities.png",
            "Average probability predicted by each model",
        )

        show_csv(
            STAGE2_XAI_DIR
            / "average_model_probabilities.csv",
            "Average model probabilities",
        )


st.info(
    "SHAP values indicate how strongly a feature or model "
    "moves a prediction away from the model's baseline. "
    "They describe model behaviour rather than proving causality."
)

