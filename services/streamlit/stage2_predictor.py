from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

# Reduce TensorFlow informational logging before it is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "stage2"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


LOW_PATTERNS = {
    "low_typo_wording": r"\b(typo|spelling|grammar|wording|text change|copy|label text|wrong text)\b",
    "low_cosmetic": r"\b(cosmetic|visual polish|polish|minor visual|small visual|aesthetic)\b",
    "low_alignment_layout": r"\b(alignment|aligned|padding|margin|spacing|layout|overlap|misaligned|truncated text)\b",
    "low_icon_color_theme": r"\b(icon|color|colour|theme|dark mode|light mode|contrast|wrong icon|missing icon)\b",
    "low_documentation": r"\b(documentation|doc|docs|help page|release note|tooltip|message wording)\b",
    "low_localization": r"\b(localization|localisation|translation|locale|l10n|rtl|language pack)\b",
    "low_preference_minor": r"\b(preference|pref|setting label|checkbox label|about:config|option label)\b",
    "low_nitpick": r"\b(nit|nitpick|minor issue|minor bug|small issue|edge case)\b",
    "low_enhancement_like": r"\b(enhancement|feature request|wishlist|nice to have|improvement)\b",
    "low_test_only": r"\b(test only|intermittent test|test failure only|ci only|lint|unit test)\b",
}

MEDIUM_PATTERNS = {
    "med_regression": r"\b(regression|regressed|worked before|used to work|since update|new regression)\b",
    "med_function_broken": r"\b(does not work|doesn't work|not working|broken|fails|failure|unable to|cannot|can't)\b",
    "med_core_workflow": r"\b(send|receive|reply|forward|compose|open message|read message|delete message|move message|search mail|download)\b",
    "med_mail_account": r"\b(imap|smtp|pop3|exchange|oauth|account|login|authentication|password|server settings)\b",
    "med_sync_calendar": r"\b(sync|synchroni[sz]e|calendar|caldav|carddav|address book|contact)\b",
    "med_attachment": r"\b(attachment|attach|detach|inline attachment|download attachment|upload attachment)\b",
    "med_folder_message": r"\b(folder|inbox|sent folder|trash|junk|spam|archive|message list|thread|conversation)\b",
    "med_data_visible_missing": r"\b(missing|disappear|not shown|not displayed|blank|empty|invisible|lost view)\b",
    "med_performance_moderate": r"\b(slow|lag|delay|timeout|timed out|performance|takes long|unresponsive for)\b",
    "med_reproducible": r"\b(steps to reproduce|str|reproduce|reproducible|actual result|expected result)\b",
    "med_platform_specific": r"\b(windows|linux|mac|android|ios|wayland|x11|ubuntu|fedora)\b",
    "med_accessibility": r"\b(accessibility|a11y|screen reader|nvda|voiceover|aria|keyboard navigation|focus)\b",
}

HIGH_RISK_PATTERNS = {
    "risk_crash": r"\b(crash|crashed|crashes|segfault|segmentation fault|fatal|abort|panic)\b",
    "risk_security": r"\b(security|vulnerab|exploit|xss|csrf|privilege|permission bypass|auth bypass|malware)\b",
    "risk_data_loss": r"\b(data loss|lost data|corrupt|corruption|deleted|permanent loss|overwrite)\b",
    "risk_hang_freeze": r"\b(hang|freeze|frozen|deadlock|not responding|lockup|shutdownhang)\b",
    "risk_oom": r"\b(oom|out of memory|memory leak|large allocation|mozalloc_abort)\b",
    "risk_startup_blocker": r"\b(fails to start|cannot start|unable to start|startup crash|crash on startup)\b",
}

ANTI_HIGH_PATTERNS = {
    "anti_not_crash": r"\b(not a crash|does not crash|doesn't crash|no crash|without crashing)\b",
    "anti_not_security": r"\b(not security|not a security issue|no security impact)\b",
    "anti_minor_language": r"\b(minor|trivial|cosmetic|typo|nit|polish|small issue)\b",
    "anti_workaround": r"\b(workaround|can work around|temporary workaround|alternative way)\b",
    "anti_low_impact": r"\b(low impact|not critical|not urgent|rare case|edge case)\b",
}

PROCESS_COLS_CANDIDATES = [
    "summary_len",
    "bug_text_len",
    "model_text_len",
    "has_bug_text",
    "has_bug_report_text",
    "has_stack_trace",
    "has_steps_to_reproduce",
]


@st.cache_resource(show_spinner="Loading Stage-2 models...")
def load_stage2_resources() -> dict[str, Any]:
    required_files = [
        "xgb_low_medium.joblib",
        "lgbm_low_medium.joblib",
        "calibrator_low_medium.joblib",
        "scaler_tree_low_medium.joblib",
        "tfidf_low_medium.joblib",
        "svd_low_medium.joblib",
        "feature_schema_low_medium.json",
        "category_rate_encoders_low_medium.joblib",
        "inference_manifest_low_medium.json",
    ]

    missing = [
        filename
        for filename in required_files
        if not (MODEL_DIR / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Stage-2 model artifacts: " + ", ".join(missing)
        )

    with open(
        MODEL_DIR / "feature_schema_low_medium.json",
        encoding="utf-8",
    ) as file:
        schema = json.load(file)

    with open(
        MODEL_DIR / "inference_manifest_low_medium.json",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    resources: dict[str, Any] = {
        "xgb": joblib.load(MODEL_DIR / "xgb_low_medium.joblib"),
        "lgbm": joblib.load(MODEL_DIR / "lgbm_low_medium.joblib"),
        "calibrator": joblib.load(
            MODEL_DIR / "calibrator_low_medium.joblib"
        ),
        "tree_scaler": joblib.load(
            MODEL_DIR / "scaler_tree_low_medium.joblib"
        ),
        "tfidf": joblib.load(MODEL_DIR / "tfidf_low_medium.joblib"),
        "svd": joblib.load(MODEL_DIR / "svd_low_medium.joblib"),
        "category_encoders": joblib.load(
            MODEL_DIR / "category_rate_encoders_low_medium.joblib"
        ),
        "schema": schema,
        "manifest": manifest,
        "encoder": SentenceTransformer(EMBEDDING_MODEL),
        "lstm_model": None,
        "lstm_tokenizer": None,
        "lstm_meta_scaler": None,
    }

    if manifest.get("lstm_used", False):
        lstm_files = [
            "lstm_low_medium.keras",
            "lstm_tokenizer_low_medium.joblib",
            "lstm_meta_scaler_low_medium.joblib",
        ]
        missing_lstm = [
            filename
            for filename in lstm_files
            if not (MODEL_DIR / filename).exists()
        ]
        if missing_lstm:
            raise FileNotFoundError(
                "Stage-2 manifest requires LSTM artifacts, but these "
                "files are missing: " + ", ".join(missing_lstm)
            )

        from tensorflow.keras.models import load_model

        resources["lstm_model"] = load_model(
            MODEL_DIR / "lstm_low_medium.keras",
            compile=False,
        )
        resources["lstm_tokenizer"] = joblib.load(
            MODEL_DIR / "lstm_tokenizer_low_medium.joblib"
        )
        resources["lstm_meta_scaler"] = joblib.load(
            MODEL_DIR / "lstm_meta_scaler_low_medium.joblib"
        )

    return resources


def clean_category_value(value: Any) -> str:
    if pd.isna(value):
        return "unknown"

    text = str(value).strip().lower()
    if text in {
        "",
        "nan",
        "none",
        "null",
        "--",
        "unknown",
        "unspecified",
    }:
        return "unknown"

    return re.sub(r"\s+", " ", text)


def component_family(value: Any) -> str:
    text = clean_category_value(value)

    if any(key in text for key in ["compose", "editor", "message compose"]):
        return "compose"
    if any(
        key in text
        for key in [
            "imap",
            "smtp",
            "pop",
            "mailnews",
            "message",
            "folder",
            "mail",
        ]
    ):
        return "mail_core"
    if any(key in text for key in ["address", "carddav", "contact"]):
        return "contacts"
    if any(key in text for key in ["calendar", "caldav", "event"]):
        return "calendar"
    if any(key in text for key in ["ui", "theme", "front", "interface", "widget"]):
        return "ui"
    if any(key in text for key in ["accessibility", "a11y"]):
        return "accessibility"
    if any(key in text for key in ["performance", "memory", "startup"]):
        return "performance"
    if any(key in text for key in ["security", "privacy", "crypto", "certificate"]):
        return "security"
    if any(key in text for key in ["build", "ci", "test", "lint"]):
        return "build_test"
    if any(key in text for key in ["devtools", "debugger", "console"]):
        return "devtools"
    if any(key in text for key in ["sync", "account"]):
        return "sync_account"

    return "other"


def _infer_text_flags(text: str) -> tuple[int, int]:
    has_stack_trace = int(
        bool(
            re.search(
                r"\b(stack trace|traceback)\b|(?:\r?\n)\s*(?:at\s+\S+|File\s+\".*\",\s+line\s+\d+)",
                text,
                flags=re.IGNORECASE,
            )
        )
    )
    has_steps = int(
        bool(
            re.search(
                r"\b(steps to reproduce|actual results?|expected results?|STR)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
    )
    return has_stack_trace, has_steps


def build_stage2_input_row(
    summary: str,
    bug_report_text: str = "",
    component: str = "UNKNOWN",
    bug_type: str = "UNKNOWN",
    version: str = "UNKNOWN",
    platform: str = "UNKNOWN",
    op_sys: str = "UNKNOWN",
    initial_priority: str = "UNKNOWN",
    creator_experience_level: str = "UNKNOWN",
) -> pd.DataFrame:
    summary = str(summary or "").strip()
    bug_report_text = str(bug_report_text or "").strip()

    if not summary and not bug_report_text:
        raise ValueError("Enter a bug summary or bug-report description.")

    text_final = (
        f"{summary} [SEP] {bug_report_text}"
        if summary and bug_report_text
        else summary or bug_report_text
    )

    has_stack_trace, has_steps = _infer_text_flags(text_final)

    priority = str(initial_priority or "UNKNOWN").strip().upper()
    if priority in {"", "NAN", "NONE"}:
        priority = "UNKNOWN"

    return pd.DataFrame(
        [
            {
                "summary": summary,
                "bug_report_text": (
                    bug_report_text if bug_report_text else None
                ),
                "text_final": text_final,
                "model_text": text_final,
                "component": str(component or "UNKNOWN"),
                "type": str(bug_type or "UNKNOWN"),
                "version": str(version or "UNKNOWN"),
                "platform": str(platform or "UNKNOWN"),
                "op_sys": str(op_sys or "UNKNOWN"),
                "creator_experience_level": str(
                    creator_experience_level or "UNKNOWN"
                ),
                "initial_priority": priority,
                "initial_priority_clean": priority,
                "summary_len": len(summary),
                "bug_text_len": len(bug_report_text),
                "model_text_len": len(text_final),
                "has_bug_text": int(bool(bug_report_text)),
                "has_bug_report_text": int(bool(bug_report_text)),
                "has_stack_trace": has_stack_trace,
                "has_steps_to_reproduce": has_steps,
                # Safe defaults when the dashboard does not collect
                # historical creator statistics.
                "creator_contributor_match": 0,
                "creator_bugs_filed_log": 0.0,
                "creator_comments_made_log": 0.0,
                "creator_patches_submitted_log": 0.0,
                "creator_patches_reviewed_log": 0.0,
                "creator_activity_score": 0.0,
                "creator_experience_rank": 0.0,
            }
        ]
    )


def add_low_medium_pattern_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    rows = df.copy()
    text = rows["text_final"].fillna("").astype(str).str.lower()
    feature_cols: list[str] = []

    for column, pattern in LOW_PATTERNS.items():
        rows[column] = text.str.contains(
            pattern,
            regex=True,
        ).astype(int)
        feature_cols.append(column)

    for column, pattern in MEDIUM_PATTERNS.items():
        rows[column] = text.str.contains(
            pattern,
            regex=True,
        ).astype(int)
        feature_cols.append(column)

    for column, pattern in HIGH_RISK_PATTERNS.items():
        rows[column] = text.str.contains(
            pattern,
            regex=True,
        ).astype(int)
        feature_cols.append(column)

    for column, pattern in ANTI_HIGH_PATTERNS.items():
        rows[column] = text.str.contains(
            pattern,
            regex=True,
        ).astype(int)
        feature_cols.append(column)

    low_cols = list(LOW_PATTERNS)
    medium_cols = list(MEDIUM_PATTERNS)
    risk_cols = list(HIGH_RISK_PATTERNS)
    anti_cols = list(ANTI_HIGH_PATTERNS)

    rows["low_pattern_count"] = rows[low_cols].sum(axis=1)
    rows["medium_pattern_count"] = rows[medium_cols].sum(axis=1)
    rows["high_risk_pattern_count"] = rows[risk_cols].sum(axis=1)
    rows["anti_high_pattern_count"] = rows[anti_cols].sum(axis=1)
    rows["no_high_risk_signal"] = (
        rows["high_risk_pattern_count"] == 0
    ).astype(int)
    rows["medium_minus_low_pattern_score"] = (
        rows["medium_pattern_count"] - rows["low_pattern_count"]
    )
    rows["low_minus_medium_pattern_score"] = (
        rows["low_pattern_count"] - rows["medium_pattern_count"]
    )

    derived_cols = [
        "low_pattern_count",
        "medium_pattern_count",
        "high_risk_pattern_count",
        "anti_high_pattern_count",
        "no_high_risk_signal",
        "medium_minus_low_pattern_score",
        "low_minus_medium_pattern_score",
    ]

    return rows, feature_cols + derived_cols


def add_metadata_process_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    rows = df.copy()
    feature_cols: list[str] = []

    rows["char_count"] = (
        rows["text_final"].fillna("").astype(str).str.len()
    )
    rows["word_count"] = (
        rows["text_final"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )
    rows["uppercase_ratio"] = rows["text_final"].apply(
        lambda value: sum(ch.isupper() for ch in str(value))
        / max(len(str(value)), 1)
    )
    rows["punctuation_density"] = rows["text_final"].apply(
        lambda value: sum(ch in "!?;:." for ch in str(value))
        / max(len(str(value)), 1)
    )
    feature_cols.extend(
        [
            "char_count",
            "word_count",
            "uppercase_ratio",
            "punctuation_density",
        ]
    )

    for column in PROCESS_COLS_CANDIDATES:
        if column in rows.columns:
            rows[column] = pd.to_numeric(
                rows[column],
                errors="coerce",
            ).fillna(0)
            feature_cols.append(column)

    categorical_cols: list[str] = []
    for column in [
        "component",
        "type",
        "version",
        "platform",
        "op_sys",
        "creator_experience_level",
    ]:
        if column in rows.columns:
            clean_column = f"{column}_clean_lm"
            rows[clean_column] = rows[column].map(clean_category_value)
            categorical_cols.append(clean_column)

            missing_column = f"{column}_missing_lm"
            rows[missing_column] = (
                rows[clean_column] == "unknown"
            ).astype(int)
            feature_cols.append(missing_column)

    if "component" in rows.columns:
        rows["component_family_lm"] = rows["component"].map(
            component_family
        )
        categorical_cols.append("component_family_lm")

    creator_numeric_cols = [
        "creator_contributor_match",
        "creator_bugs_filed_log",
        "creator_comments_made_log",
        "creator_patches_submitted_log",
        "creator_patches_reviewed_log",
        "creator_activity_score",
        "creator_experience_rank",
    ]
    for column in creator_numeric_cols:
        if column in rows.columns:
            rows[column] = pd.to_numeric(
                rows[column],
                errors="coerce",
            ).fillna(0)
            feature_cols.append(column)

    if "initial_priority_clean" in rows.columns:
        rows["initial_priority_clean_lm"] = (
            rows["initial_priority_clean"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"": "UNKNOWN"})
        )
    elif "initial_priority" in rows.columns:
        rows["initial_priority_clean_lm"] = (
            rows["initial_priority"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"": "UNKNOWN"})
        )
    else:
        rows["initial_priority_clean_lm"] = "UNKNOWN"

    priority_rank_map = {
        "P1": 1,
        "P2": 2,
        "P3": 3,
        "P4": 4,
        "P5": 5,
        "--": 99,
        "UNKNOWN": 99,
    }
    rows["initial_priority_rank_lm"] = (
        rows["initial_priority_clean_lm"]
        .map(priority_rank_map)
        .fillna(99)
        .astype(float)
    )
    rows["initial_priority_is_p3_or_lower_lm"] = (
        rows["initial_priority_clean_lm"]
        .isin(["P3", "P4", "P5"])
        .astype(int)
    )
    rows["initial_priority_is_unset_lm"] = (
        rows["initial_priority_clean_lm"]
        .isin(["--", "UNKNOWN"])
        .astype(int)
    )

    feature_cols.extend(
        [
            "initial_priority_rank_lm",
            "initial_priority_is_p3_or_lower_lm",
            "initial_priority_is_unset_lm",
        ]
    )
    categorical_cols.append("initial_priority_clean_lm")

    return rows, feature_cols, categorical_cols


def apply_target_rate_weights(
    df: pd.DataFrame,
    encoders: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    rows = df.copy()
    new_cols: list[str] = []

    for column, encoder in encoders.items():
        if column not in rows.columns:
            rows[column] = "unknown"

        keys = rows[column].map(clean_category_value)

        rate_column = f"{column}_medium_rate_weight"
        lift_column = f"{column}_medium_lift_weight"
        frequency_column = f"{column}_freq_log_weight"

        rows[rate_column] = (
            keys.map(encoder["mapping_rate"])
            .fillna(encoder["global_rate"])
            .astype(float)
        )
        rows[lift_column] = (
            keys.map(encoder["mapping_lift"])
            .fillna(0.0)
            .astype(float)
        )
        rows[frequency_column] = (
            keys.map(encoder["mapping_freq"])
            .fillna(0.0)
            .astype(float)
        )

        new_cols.extend(
            [rate_column, lift_column, frequency_column]
        )

    return rows, new_cols


def prepare_stage2_features(
    input_df: pd.DataFrame,
    resources: dict[str, Any],
) -> pd.DataFrame:
    rows, _ = add_low_medium_pattern_features(input_df)
    rows, _, _ = add_metadata_process_features(rows)
    rows, _ = apply_target_rate_weights(
        rows,
        resources["category_encoders"],
    )
    return rows


def build_stage2_tree_matrix(
    prepared_rows: pd.DataFrame,
    resources: dict[str, Any],
) -> np.ndarray:
    feature_cols = resources["manifest"]["tree_feature_cols"]

    rows = prepared_rows.copy()
    for column in feature_cols:
        if column not in rows.columns:
            rows[column] = 0

    X_tfidf = resources["tfidf"].transform(rows["text_final"])
    X_svd = resources["svd"].transform(X_tfidf)

    X_embedding = resources["encoder"].encode(
        rows["text_final"].tolist(),
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)

    X_metadata = resources["tree_scaler"].transform(
        rows[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy()
    )

    matrix = np.hstack(
        [X_embedding, X_svd, X_metadata]
    ).astype(np.float32)

    expected_features = resources["xgb"].n_features_in_
    if matrix.shape[1] != expected_features:
        raise ValueError(
            "Stage-2 tree feature mismatch: "
            f"created {matrix.shape[1]}, expected {expected_features}."
        )

    return matrix


def _humanise_stage2_signal(name: str) -> str:
    cleaned = name

    for prefix in [
        "low_",
        "med_",
        "risk_",
        "anti_",
    ]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]

    return cleaned.replace("_", " ").strip().title()


def get_stage2_active_signals(
    prepared_rows: pd.DataFrame,
) -> dict[str, list[str]]:
    row = prepared_rows.iloc[0]

    def active_from(patterns: dict[str, str]) -> list[str]:
        return [
            _humanise_stage2_signal(column)
            for column in patterns
            if column in prepared_rows.columns
            and int(row[column]) == 1
        ]

    return {
        "low_evidence": active_from(LOW_PATTERNS),
        "medium_evidence": active_from(MEDIUM_PATTERNS),
        "high_risk_evidence": active_from(
            HIGH_RISK_PATTERNS
        ),
        "anti_high_evidence": active_from(
            ANTI_HIGH_PATTERNS
        ),
    }



def predict_stage2(
    summary: str,
    bug_report_text: str = "",
    component: str = "UNKNOWN",
    bug_type: str = "UNKNOWN",
    version: str = "UNKNOWN",
    platform: str = "UNKNOWN",
    op_sys: str = "UNKNOWN",
    initial_priority: str = "UNKNOWN",
    creator_experience_level: str = "UNKNOWN",
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if resources is None:
        resources = load_stage2_resources()

    input_row = build_stage2_input_row(
        summary=summary,
        bug_report_text=bug_report_text,
        component=component,
        bug_type=bug_type,
        version=version,
        platform=platform,
        op_sys=op_sys,
        initial_priority=initial_priority,
        creator_experience_level=creator_experience_level,
    )

    prepared_rows = prepare_stage2_features(
        input_row,
        resources,
    )
    tree_matrix = build_stage2_tree_matrix(
        prepared_rows,
        resources,
    )

    xgb_probability = float(
        resources["xgb"].predict_proba(tree_matrix)[0, 1]
    )
    lgbm_probability = float(
        resources["lgbm"].predict_proba(tree_matrix)[0, 1]
    )

    model_probabilities: dict[str, float] = {
        "xgb_prob_medium": xgb_probability,
        "lgbm_prob_medium": lgbm_probability,
    }

    lstm_probability: float | None = None
    if resources["manifest"].get("lstm_used", False):
        from tensorflow.keras.preprocessing.sequence import (
            pad_sequences,
        )

        lstm_feature_cols = resources["manifest"][
            "lstm_feature_cols"
        ]
        lstm_rows = prepared_rows.copy()

        for column in lstm_feature_cols:
            if column not in lstm_rows.columns:
                lstm_rows[column] = 0

        sequences = resources["lstm_tokenizer"].texts_to_sequences(
            lstm_rows["text_final"].astype(str).tolist()
        )
        padded = pad_sequences(
            sequences,
            maxlen=int(
                resources["manifest"].get("lstm_max_len", 180)
            ),
            padding=resources["manifest"].get(
                "lstm_padding",
                "post",
            ),
            truncating=resources["manifest"].get(
                "lstm_truncating",
                "post",
            ),
        )

        lstm_metadata = resources[
            "lstm_meta_scaler"
        ].transform(
            lstm_rows[lstm_feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy()
        ).astype(np.float32)

        expected_meta_width = int(
            resources["lstm_model"].input_shape[1][-1]
        )
        if lstm_metadata.shape[1] != expected_meta_width:
            raise ValueError(
                "Stage-2 LSTM metadata mismatch: "
                f"created {lstm_metadata.shape[1]}, "
                f"expected {expected_meta_width}."
            )

        lstm_probability = float(
            resources["lstm_model"].predict(
                [padded, lstm_metadata],
                verbose=0,
            ).ravel()[0]
        )
        model_probabilities["lstm_prob_medium"] = (
            lstm_probability
        )

    stack_feature_cols = resources["manifest"][
        "stack_feature_cols"
    ]
    missing_stack = [
        column
        for column in stack_feature_cols
        if column not in model_probabilities
    ]
    if missing_stack:
        raise ValueError(
            "Missing Stage-2 stacked-model inputs: "
            + ", ".join(missing_stack)
        )

    stack_input = pd.DataFrame(
        [
            {
                column: model_probabilities[column]
                for column in stack_feature_cols
            }
        ]
    )

    active_signals = get_stage2_active_signals(
        prepared_rows
    )

    calibrator = resources["calibrator"]

    coefficients = np.asarray(
        calibrator.coef_
    ).reshape(-1)

    stack_log_odds_contributions = {
        column: float(
            stack_input.iloc[0][column]
            * coefficients[index]
        )
        for index, column in enumerate(
            stack_feature_cols
        )
    }

    stack_intercept_log_odds = float(
        np.asarray(
            calibrator.intercept_
        ).reshape(-1)[0]
    )

    probability_medium = float(
        resources["calibrator"].predict_proba(stack_input)[0, 1]
    )

    threshold_info = resources["manifest"]["best_threshold"]
    if isinstance(threshold_info, dict):
        threshold = float(threshold_info["threshold"])
    else:
        threshold = float(threshold_info)

    is_medium = probability_medium >= threshold
    label = "MEDIUM" if is_medium else "LOW"
    confidence = (
        probability_medium if is_medium else 1.0 - probability_medium
    )

    return {
        "label": label,
        "probability_medium": probability_medium,
        "confidence": float(confidence),
        "threshold": threshold,
        "xgb_probability_medium": xgb_probability,
        "lgbm_probability_medium": lgbm_probability,
        "lstm_probability_medium": lstm_probability,
        "stack_feature_cols": stack_feature_cols,
         "active_signals": active_signals,
        "model_probabilities": model_probabilities,
        "stack_log_odds_contributions": (
            stack_log_odds_contributions
        ),
        "stack_intercept_log_odds": (
            stack_intercept_log_odds
        ),
    }
