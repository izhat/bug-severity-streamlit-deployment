
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Model Performance",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

STAGE1_DIR = PROJECT_ROOT / "models" / "stage1"
STAGE1_RESULTS = STAGE1_DIR / "aggressive"

STAGE2_DIR = (
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
        dataframe = pd.read_csv(path, low_memory=False)
        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning(f"Missing file: {path.name}")


st.title("Model Performance")
st.caption(
    "Cross-project evaluation results for Firefox → Thunderbird"
)

stage1_tab, stage2_tab = st.tabs(
    [
        "Stage-1: HIGH vs NOT_HIGH",
        "Stage-2: LOW vs MEDIUM",
    ]
)


# ============================================================
# STAGE-1
# ============================================================

with stage1_tab:
    st.header("Stage-1: HIGH vs NOT_HIGH")

    st.write(
        "Stage-1 prioritises detection of HIGH-severity bugs "
        "before passing NOT_HIGH reports to Stage-2."
    )

    summary_path = STAGE1_RESULTS / "config_and_summary.json"

    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as file:
            stage1_summary = json.load(file)

        with st.expander("Stage-1 experiment summary"):
            st.json(stage1_summary)

    st.subheader("Performance across label-audit filters")

    chart_options = {
        "Accuracy": STAGE1_DIR / "accuracy_across_filters.png",
        "Macro-F1": STAGE1_DIR / "macro_f1_across_filters.png",
        "HIGH F1": STAGE1_DIR / "high_f1_across_filters.png",
        "HIGH precision": (
            STAGE1_DIR / "high_precision_across_filters.png"
        ),
        "HIGH recall": (
            STAGE1_DIR / "high_recall_across_filters.png"
        ),
        "Error counts": (
            STAGE1_DIR / "error_counts_across_filters.png"
        ),
    }

    selected_chart = st.selectbox(
        "Select Stage-1 comparison graph",
        options=list(chart_options.keys()),
    )

    show_image(
        chart_options[selected_chart],
        selected_chart,
    )

    st.subheader("Strict Thunderbird test results")

    col1, col2 = st.columns([1, 1])

    with col1:
        show_image(
            STAGE1_RESULTS
            / "confusion_matrix_aggressive.png",
            "Stage-1 confusion matrix",
        )

    with col2:
        show_image(
            STAGE1_RESULTS
            / "error_analysis_report"
            / "stage1_error_summary.png",
            "Stage-1 error summary",
        )

    show_csv(
        STAGE1_RESULTS / "metrics.csv",
        "Stage-1 evaluation metrics",
    )

    show_csv(
        STAGE1_RESULTS
        / "v4_xgb_probability_blend_weight_sweep.csv",
        "V4 and XGBoost probability-blend search",
    )


# ============================================================
# STAGE-2
# ============================================================

with stage2_tab:
    st.header("Stage-2: LOW vs MEDIUM")

    st.write(
        "Stage-2 classifies reports that Stage-1 predicts "
        "as NOT_HIGH."
    )

    summary_path = (
        STAGE2_DIR / "config_and_summary_low_medium.json"
    )

    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as file:
            stage2_summary = json.load(file)

        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric(
            "Accuracy",
            f"{stage2_summary.get('accuracy', 0):.2%}",
        )

        metric2.metric(
            "Macro-F1",
            f"{stage2_summary.get('macro_f1', 0):.3f}",
        )

        metric3.metric(
            "LOW F1",
            f"{stage2_summary.get('low_f1', 0):.3f}",
        )

        metric4.metric(
            "MEDIUM F1",
            f"{stage2_summary.get('medium_f1', 0):.3f}",
        )

    st.subheader("Strict Thunderbird test performance")

    col1, col2 = st.columns(2)

    with col1:
        show_image(
            STAGE2_DIR / "confusion_matrix_low_medium.png",
            "Stage-2 LOW vs MEDIUM confusion matrix",
        )

    with col2:
        show_image(
            STAGE2_DIR / "stage2_ablation_macro_f1.png",
            "Stage-2 model ablation comparison",
        )

    st.subheader("Threshold tuning")

    show_image(
        STAGE2_DIR / "stage2_threshold_tuning_curve.png",
        "Stage-2 MEDIUM threshold tuning",
    )

    st.subheader("Ensemble model contribution")

    col1, col2 = st.columns(2)

    with col1:
        show_image(
            STAGE2_DIR
            / "shap_ensemble_model_contribution.png",
            "Global contribution of XGBoost, LightGBM and LSTM",
        )

    with col2:
        show_image(
            STAGE2_DIR / "average_model_probabilities.png",
            "Average model probability comparison",
        )

    show_csv(
        STAGE2_DIR / "metrics_low_medium.csv",
        "Stage-2 validation and strict-test metrics",
    )

    show_csv(
        STAGE2_DIR / "stage2_ablation_results.csv",
        "Stage-2 model ablation results",
    )

