
from __future__ import annotations

import streamlit as st

from stage1_predictor import predict_stage1
from stage2_predictor import predict_stage2
import altair as alt 
import pandas as pd

def render_live_explanation(
    stage1_result: dict,
    stage2_result: dict | None,
) -> None:
    st.divider()
    st.subheader("Why this prediction?")

    # ========================================================
    # STAGE-1 LOCAL EXPLANATION
    # ========================================================

    st.markdown("### Stage-1 explanation")

    stage1_contributions = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Contribution": contribution,
            }
            for model_name, contribution
            in stage1_result["model_contributions"].items()
        ]
    )

    stage1_chart = (
        alt.Chart(stage1_contributions)
        .mark_bar()
        .encode(
            x=alt.X(
                "Contribution:Q",
                title="Contribution to HIGH probability",
                axis=alt.Axis(format=".0%"),
            ),
            y=alt.Y(
                "Model:N",
                sort="-x",
                title=None,
            ),
            tooltip=[
                "Model:N",
                alt.Tooltip(
                    "Contribution:Q",
                    format=".2%",
                ),
            ],
        )
        .properties(height=180)
    )

    st.altair_chart(
        stage1_chart,
        use_container_width=True,
    )

    active_stage1_signals = stage1_result.get(
        "active_signals",
        [],
    )

    if active_stage1_signals:
        st.markdown(
            "**Detected Stage-1 signals:** "
            + ", ".join(active_stage1_signals)
        )
    else:
        st.caption(
            "No explicit GraphLens severity signal was detected."
        )

    # ========================================================
    # STAGE-2 LOCAL EXPLANATION
    # ========================================================

    if stage2_result is None:
        st.info(
            "Stage-2 was not executed because Stage-1 "
            "classified this report as HIGH."
        )
        return

    st.markdown("### Stage-2 explanation")

    stage2_signals = stage2_result.get(
        "active_signals",
        {},
    )

    signal_col1, signal_col2 = st.columns(2)

    with signal_col1:
        medium_evidence = stage2_signals.get(
            "medium_evidence",
            [],
        )

        st.markdown("**Evidence supporting MEDIUM**")

        if medium_evidence:
            for signal in medium_evidence:
                st.write(f"• {signal}")
        else:
            st.caption("No explicit MEDIUM pattern detected.")

    with signal_col2:
        low_evidence = stage2_signals.get(
            "low_evidence",
            [],
        )

        st.markdown("**Evidence supporting LOW**")

        if low_evidence:
            for signal in low_evidence:
                st.write(f"• {signal}")
        else:
            st.caption("No explicit LOW pattern detected.")

    additional_signals = (
        stage2_signals.get("high_risk_evidence", [])
        + stage2_signals.get("anti_high_evidence", [])
    )

    if additional_signals:
        st.markdown(
            "**Additional detected signals:** "
            + ", ".join(additional_signals)
        )

    log_odds_contributions = stage2_result.get(
        "stack_log_odds_contributions",
        {},
    )

    contribution_labels = {
        "xgb_prob_medium": "XGBoost",
        "lgbm_prob_medium": "LightGBM",
        "lstm_prob_medium": "LSTM",
    }

    contribution_df = pd.DataFrame(
        [
            {
                "Model": contribution_labels.get(
                    model_name,
                    model_name,
                ),
                "Log-odds contribution": contribution,
                "Direction": (
                    "Toward MEDIUM"
                    if contribution >= 0
                    else "Toward LOW"
                ),
            }
            for model_name, contribution
            in log_odds_contributions.items()
        ]
    )

    stage2_chart = (
        alt.Chart(contribution_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Log-odds contribution:Q",
                title="Contribution to final Stage-2 decision",
            ),
            y=alt.Y(
                "Model:N",
                sort="-x",
                title=None,
            ),
            color=alt.Color(
                "Direction:N",
                title="Decision direction",
            ),
            tooltip=[
                "Model:N",
                "Direction:N",
                alt.Tooltip(
                    "Log-odds contribution:Q",
                    format=".4f",
                ),
            ],
        )
        .properties(height=210)
    )

    st.altair_chart(
        stage2_chart,
        use_container_width=True,
    )

    st.caption(
        "Positive values move the stacked prediction toward "
        "MEDIUM; negative values move it toward LOW."
    )

    intercept = stage2_result.get(
        "stack_intercept_log_odds",
    )

    if intercept is not None:
        st.metric(
            "Stacked-model baseline log-odds",
            f"{intercept:.4f}",
        )

    st.caption(
        "Detected signals provide a human-readable interpretation. "
        "The contribution graph shows the exact inputs used by the "
        "final stacked Stage-2 calibrator."
    )








st.set_page_config(
    page_title="Bug Severity Dashboard",
    layout="wide",
)

st.title("Bug Severity Dashboard")
st.caption(
    "Explainable within-domain cross-project bug severity prediction"
)

st.subheader("Bug severity prediction")
st.write(
    "Stage-1 identifies HIGH-severity bugs. "
    "Reports classified as NOT_HIGH are passed to Stage-2 "
    "for LOW versus MEDIUM classification."
)

with st.form("severity_prediction_form"):
    summary = st.text_area(
        "Bug summary",
        value="Typo in preferences button label",
        height=90,
    )

    bug_report_text = st.text_area(
        "Bug description",
        value=(
            "The button text is misspelled, but all application "
            "functions work normally."
        ),
        height=150,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        component = st.text_input(
            "Component",
            value="General",
        )

        bug_type = st.selectbox(
            "Bug type",
            options=[
                "defect",
                "enhancement",
                "task",
                "UNKNOWN",
            ],
        )

    with col2:
        platform = st.selectbox(
            "Platform",
            options=[
                "Desktop",
                "All",
                "Mobile",
                "UNKNOWN",
            ],
        )

        op_sys = st.selectbox(
            "Operating system",
            options=[
                "Windows",
                "Linux",
                "macOS",
                "All",
                "UNKNOWN",
            ],
        )

    with col3:
        initial_priority = st.selectbox(
            "Initial priority",
            options=[
                "UNKNOWN",
                "P1",
                "P2",
                "P3",
                "P4",
                "P5",
                "--",
            ],
        )

    predict_button = st.form_submit_button(
        "Predict severity",
        type="primary",
    )


if predict_button:
    try:
        with st.spinner("Running Stage-1 HIGH vs NOT_HIGH model..."):
            stage1_result = predict_stage1(
                summary=summary,
                bug_report_text=bug_report_text,
                component=component,
                bug_type=bug_type,
                platform=platform,
                op_sys=op_sys,
                initial_priority=initial_priority,
            )

        stage2_result = None

        if stage1_result["label"] == "HIGH":
            final_label = "HIGH"
            final_confidence = stage1_result["confidence"]
        else:
            with st.spinner("Running Stage-2 LOW vs MEDIUM model..."):
                stage2_result = predict_stage2(
                    summary=summary,
                    bug_report_text=bug_report_text,
                    component=component,
                    bug_type=bug_type,
                    platform=platform,
                    op_sys=op_sys,
                    initial_priority=initial_priority,
                )

            final_label = stage2_result["label"]
            final_confidence = stage2_result["confidence"]

        if final_label == "HIGH":
            st.error("Final predicted severity: HIGH")
        elif final_label == "MEDIUM":
            st.warning("Final predicted severity: MEDIUM")
        else:
            st.success("Final predicted severity: LOW")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Final severity",
            final_label,
        )

        metric2.metric(
            "Stage-1 HIGH probability",
            f"{stage1_result['probability_high']:.2%}",
        )

        metric3.metric(
            "Final prediction confidence",
            f"{final_confidence:.2%}",
        )

        st.progress(
            min(
                max(
                    float(stage1_result["probability_high"]),
                    0.0,
                ),
                1.0,
            ),
            text=(
                "Stage-1 HIGH probability: "
                f"{stage1_result['probability_high']:.2%}"
            ),
        )

        if stage2_result is not None:
            st.progress(
                min(
                    max(
                        float(
                            stage2_result["probability_medium"]
                        ),
                        0.0,
                    ),
                    1.0,
                ),
                text=(
                    "Stage-2 MEDIUM probability: "
                    f"{stage2_result['probability_medium']:.2%}"
                ),
            )
        
        render_live_explanation( stage1_result, stage2_result, )

        with st.expander("Stage-1 calculation details"):
            st.json(
                {
                    "Stage-1 prediction": stage1_result["label"],
                    "HIGH probability": round(
                        stage1_result["probability_high"],
                        4,
                    ),
                    "V4 probability": round(
                        stage1_result["v4_probability_high"],
                        4,
                    ),
                    "XGBoost probability": round(
                        stage1_result["xgb_probability_high"],
                        4,
                    ),
                    "V4 blend weight": stage1_result[
                        "blend_weight_v4"
                    ],
                    "XGBoost blend weight": stage1_result[
                        "blend_weight_xgb"
                    ],
                    "Decision threshold": stage1_result[
                        "threshold"
                    ],
                    "Priority category": stage1_result[
                        "priority_category"
                    ],
                }
            )

        if stage2_result is not None:
            with st.expander("Stage-2 calculation details"):
                st.json(
                    {
                        "Stage-2 prediction": stage2_result[
                            "label"
                        ],
                        "MEDIUM probability": round(
                            stage2_result[
                                "probability_medium"
                            ],
                            4,
                        ),
                        "XGBoost probability": round(
                            stage2_result[
                                "xgb_probability_medium"
                            ],
                            4,
                        ),
                        "LightGBM probability": round(
                            stage2_result[
                                "lgbm_probability_medium"
                            ],
                            4,
                        ),
                        "LSTM probability": round(
                            stage2_result[
                                "lstm_probability_medium"
                            ],
                            4,
                        ),
                        "Decision threshold": stage2_result[
                            "threshold"
                        ],
                        "Stacked model inputs": stage2_result[
                            "stack_feature_cols"
                        ],
                    }
                )

        st.caption(
            "Prediction path: "
            + (
                "Stage-1 → HIGH"
                if stage1_result["label"] == "HIGH"
                else f"Stage-1 → NOT_HIGH → Stage-2 → {final_label}"
            )
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


st.divider()

st.subheader("Planned dashboard sections")
st.markdown(
    """
- Dataset overview
- Model comparison
- Explainability plots
- Cross-project performance
- Error analysis
"""
)
