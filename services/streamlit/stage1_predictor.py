from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "stage1" / "model_artifacts"

BLEND_WEIGHT_V4 = 0.35
BLEND_WEIGHT_XGB = 0.65
DECISION_THRESHOLD = 0.56

MAIN_XGB_WEIGHT = 0.60
MAIN_LGBM_WEIGHT = 0.40

CATEGORY_SUBMODEL_WEIGHT = {
    "EXCLUDE_INITIAL_UNSET": 0.70,
    "EXCLUDE_TRIAGE_ASSIGNED_LATER": 0.50,
    "NO_PRIORITY_AVAILABLE": 0.50,
}
DEFAULT_SUBMODEL_WEIGHT = 0.50

warnings.filterwarnings(
    "ignore",
    message="This pattern is interpreted as a regular expression.*",
)


GRAPH_PATTERNS = {
    "sig_crash": r"\b(crash|crashed|crashes|segfault|segmentation fault|fatal|abort|panic)\b",
    "sig_startup_crash": r"\b(startup crash|crash on startup|crashes on startup|starting up.*crash)\b",
    "sig_crash_signature": r"\b(crash in \[@|\[@ .*\]|shutdownhang|mozalloc_abort|stackoverflow|stack overflow)\b",
    "sig_oom": r"\b(oom|out of memory|large allocation|mozalloc_abort)\b",
    "sig_exception": r"\b(exception|null pointer|npe|stack trace|traceback|assertion|error)\b",
    "sig_security": r"\b(security|vulnerab|exploit|xss|csrf|permission|privilege|auth|certificate|virus|worm|malware)\b",
    "sig_data_loss": r"\b(data loss|lost data|corrupt|corruption|overwrite|deleted|missing data|bad offsets|vanish|vanished|lost|missing email|missing attachment)\b",
    "sig_hang": r"\b(hang|freeze|frozen|deadlock|not responding|stuck|lockup|shutdownhang)\b",
    "sig_memory": r"\b(memory|oom|out of memory|leak|heap|gc|mozalloc_abort)\b",
    "sig_regression": r"\b(regression|regressed|worked before|previous version|since update)\b",
    "sig_performance": r"\b(slow|performance|latency|timeout|timed out|lag|delay)\b",
    "sig_blocking": r"\b(blocker|blocks|cannot use|unusable|prevent|fails to start|stops working|cannot start|unable to start|cannot send|unable to send)\b",
    "sig_steps": r"\b(steps to reproduce|STR|reproduce|actual result|expected result)\b",
    "sig_ui_minor": r"\b(ui|cosmetic|typo|alignment|layout|icon|button|label|color)\b",
    "sig_enhancement": r"\b(enhancement|feature request|improvement|wishlist)\b",
    "sig_address_book": r"\b(vcard|carddav|address book|addressbook|contact|cardav|apple contacts)\b",
    "sig_message_list": r"\b(message list|messagelist|focus event|selection|unified folder|thread|table view)\b",
    "sig_accessibility": r"\b(screen reader|a11y|accessibility|voiceover|nvda|aria|announce|focus state|wcag|jaws)\b",
    "sig_addon": r"\b(add-on|addon|extension|xpi|verify addon|addon verification)\b",
    "sig_account": r"\b(account|imap|smtp|exchange|exquilla|account central|account creation)\b",
    "sig_attachment": r"\b(attachment|detach|inline attachment|attach file|detach all|pptx|zip attachment)\b",
    "sig_compose": r"\b(compose|reply|send|subject field|outbox|unsent|draft)\b",
    "sig_folder": r"\b(folder|inbox|spam|junk|trash|move message|delete message)\b",
    "sig_spell_check": r"\b(spell|spellcheck|inline spell|nsieditor|getinlinespellc)\b",
    "sig_html_render": r"\b(html|render|mime|html virus|mixed content|button tag|anchor tag)\b",
    "sig_startup": r"\b(startup|start up|blank.*start|3pane|account central.*start)\b",
    "sig_keyboard": r"\b(keyboard shortcut|keybinding|propagat|hotkey)\b",
    "sig_qr_code": r"\b(qr code|qrcode|high contrast|inverted color)\b",
}

PROCESS_COLS_CANDIDATES = [
    "summary_length",
    "summary_len",
    "bug_text_len",
    "model_text_len",
    "has_bug_text",
    "has_bug_report_text",
    "has_stack_trace",
    "has_steps_to_reproduce",
    "is_ci_log_like",
]

XGB_BLEND_SIGNAL_PATTERNS = {
    "sig_crash_xgb": r"\b(crash|crashed|crashes|segfault|segmentation fault|fatal|abort|panic)\b",
    "sig_startup_crash_xgb": r"\b(startup crash|crash on startup|crashes on startup)\b",
    "sig_oom_xgb": r"\b(oom|out of memory|large allocation|mozalloc_abort)\b",
    "sig_exception_xgb": r"\b(exception|null pointer|npe|stack trace|traceback|assertion)\b",
    "sig_security_xgb": r"\b(security|vulnerab|exploit|xss|csrf|permission|privilege|auth|certificate)\b",
    "sig_data_loss_xgb": r"\b(data loss|lost data|corrupt|corruption|overwrite|deleted|missing data|lost|missing email)\b",
    "sig_hang_xgb": r"\b(hang|freeze|frozen|deadlock|not responding|stuck|lockup)\b",
    "sig_memory_xgb": r"\b(memory|leak|heap|gc|out of memory|oom)\b",
    "sig_regression_xgb": r"\b(regression|regressed|worked before|previous version|since update)\b",
    "sig_performance_xgb": r"\b(slow|performance|latency|timeout|timed out|lag|delay)\b",
    "sig_blocking_xgb": r"\b(blocker|blocks|cannot use|unusable|prevent|fails to start|cannot start|cannot send)\b",
    "sig_steps_xgb": r"\b(steps to reproduce|str|reproduce|actual result|expected result)\b",
    "sig_ui_minor_xgb": r"\b(ui|cosmetic|typo|alignment|layout|icon|button|label|color)\b",
    "sig_accessibility_xgb": r"\b(screen reader|a11y|accessibility|voiceover|nvda|aria|wcag|jaws)\b",
    "sig_attachment_xgb": r"\b(attachment|detach|inline attachment|attach file|zip attachment)\b",
    "sig_compose_xgb": r"\b(compose|reply|send|subject field|outbox|unsent|draft)\b",
    "sig_folder_xgb": r"\b(folder|inbox|spam|junk|trash|move message|delete message)\b",
}


@st.cache_resource(show_spinner="Loading Stage-1 models...")
def load_stage1_resources() -> dict[str, Any]:
    required_files = [
        "xgb_main.joblib",
        "lgbm_main.joblib",
        "platt_calibrator.joblib",
        "tfidf.joblib",
        "svd.joblib",
        "scaler.joblib",
        "feature_schema.json",
        "xgb_baseline_for_probability_blend.joblib",
        "xgb_blend_tfidf.joblib",
        "xgb_blend_svd.joblib",
        "xgb_blend_scaler.joblib",
        "xgb_blend_ohe.joblib",
        "xgb_blend_feature_schema.json",
    ]
    missing = [name for name in required_files if not (MODEL_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage-1 model artifacts: " + ", ".join(missing)
        )

    with open(MODEL_DIR / "feature_schema.json", encoding="utf-8") as file:
        v4_schema = json.load(file)

    with open(
        MODEL_DIR / "xgb_blend_feature_schema.json",
        encoding="utf-8",
    ) as file:
        xgb_schema = json.load(file)

    resources: dict[str, Any] = {
        "xgb_main": joblib.load(MODEL_DIR / "xgb_main.joblib"),
        "lgbm_main": joblib.load(MODEL_DIR / "lgbm_main.joblib"),
        "calibrator": joblib.load(MODEL_DIR / "platt_calibrator.joblib"),
        "tfidf": joblib.load(MODEL_DIR / "tfidf.joblib"),
        "svd": joblib.load(MODEL_DIR / "svd.joblib"),
        "scaler": joblib.load(MODEL_DIR / "scaler.joblib"),
        "v4_schema": v4_schema,
        "xgb_blend_model": joblib.load(
            MODEL_DIR / "xgb_baseline_for_probability_blend.joblib"
        ),
        "xgb_blend_tfidf": joblib.load(
            MODEL_DIR / "xgb_blend_tfidf.joblib"
        ),
        "xgb_blend_svd": joblib.load(
            MODEL_DIR / "xgb_blend_svd.joblib"
        ),
        "xgb_blend_scaler": joblib.load(
            MODEL_DIR / "xgb_blend_scaler.joblib"
        ),
        "xgb_blend_ohe": joblib.load(
            MODEL_DIR / "xgb_blend_ohe.joblib"
        ),
        "xgb_schema": xgb_schema,
        "encoder": SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
    }

    xgb_ns_path = MODEL_DIR / "xgb_no_signal_submodel.joblib"
    lgbm_ns_path = MODEL_DIR / "lgbm_no_signal_submodel.joblib"

    resources["xgb_no_signal"] = (
        joblib.load(xgb_ns_path) if xgb_ns_path.exists() else None
    )
    resources["lgbm_no_signal"] = (
        joblib.load(lgbm_ns_path) if lgbm_ns_path.exists() else None
    )

    return resources


def _normalise_priority(value: str | None) -> str:
    priority = str(value or "UNKNOWN").strip().upper()
    if priority in {"", "NAN", "NONE"}:
        return "UNKNOWN"
    return priority


def _infer_text_flags(text: str) -> tuple[int, int, int]:
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
    is_ci_log_like = int(
        bool(
            re.search(
                r"\b(ci|continuous integration|build failed|test failed|job failed|pipeline failed)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
    )
    return has_stack_trace, has_steps, is_ci_log_like


def build_input_row(
    summary: str,
    bug_report_text: str = "",
    component: str = "UNKNOWN",
    bug_type: str = "UNKNOWN",
    platform: str = "UNKNOWN",
    op_sys: str = "UNKNOWN",
    initial_priority: str = "UNKNOWN",
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

    has_stack_trace, has_steps, is_ci_log_like = _infer_text_flags(
        text_final
    )
    priority = _normalise_priority(initial_priority)

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
                "platform": str(platform or "UNKNOWN"),
                "op_sys": str(op_sys or "UNKNOWN"),
                "initial_priority": priority,
                "initial_priority_clean": priority,
                "summary_length": len(summary),
                "summary_len": len(summary),
                "bug_text_len": len(bug_report_text),
                "model_text_len": len(text_final),
                "has_bug_text": int(bool(bug_report_text)),
                "has_bug_report_text": int(bool(bug_report_text)),
                "has_stack_trace": has_stack_trace,
                "has_steps_to_reproduce": has_steps,
                "is_ci_log_like": is_ci_log_like,
            }
        ]
    )


def add_graphlens_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    rows = df.copy()
    text = rows["text_final"].fillna("").astype(str).str.lower()

    for column, pattern in GRAPH_PATTERNS.items():
        rows[column] = text.str.contains(pattern, regex=True).astype(int)

    signal_cols = list(GRAPH_PATTERNS.keys())
    rows["graphlens_signal_count"] = rows[signal_cols].sum(axis=1)
    rows["char_count"] = rows["text_final"].str.len()
    rows["word_count"] = rows["text_final"].str.split().str.len()
    rows["uppercase_ratio"] = rows["text_final"].apply(
        lambda value: sum(ch.isupper() for ch in str(value))
        / max(len(str(value)), 1)
    )
    rows["punctuation_density"] = rows["text_final"].apply(
        lambda value: sum(ch in "!?;:." for ch in str(value))
        / max(len(str(value)), 1)
    )

    return rows, signal_cols + [
        "graphlens_signal_count",
        "char_count",
        "word_count",
        "uppercase_ratio",
        "punctuation_density",
    ]


def add_process_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    rows = df.copy()
    used_cols: list[str] = []
    safe_binary_cols = {
        "has_bug_text",
        "has_bug_report_text",
        "has_stack_trace",
        "has_steps_to_reproduce",
        "is_ci_log_like",
    }

    if "bug_report_text" in rows.columns:
        rows["bug_report_text_missing"] = (
            rows["bug_report_text"].isna().astype(int)
        )
        used_cols.append("bug_report_text_missing")

    if "bug_report_text" in rows.columns and "model_text" in rows.columns:
        rows["text_is_summary_only"] = (
            rows["bug_report_text"].isna()
            & rows["model_text"]
            .fillna("")
            .astype(str)
            .str.len()
            .between(10, 300)
        ).astype(int)
        used_cols.append("text_is_summary_only")
    elif "bug_report_text" in rows.columns:
        rows["text_is_summary_only"] = (
            rows["bug_report_text"].isna()
            & rows["text_final"]
            .fillna("")
            .astype(str)
            .str.len()
            .between(10, 300)
        ).astype(int)
        used_cols.append("text_is_summary_only")

    for column in PROCESS_COLS_CANDIDATES:
        if column not in rows.columns:
            continue

        rows[column] = pd.to_numeric(
            rows[column],
            errors="coerce",
        ).fillna(0)

        if column not in safe_binary_cols and not column.endswith("_log"):
            new_column = f"{column}_safe_log"
            rows[new_column] = np.log1p(
                np.maximum(rows[column].to_numpy(), 0)
            )
            used_cols.append(new_column)
        else:
            used_cols.append(column)

    return rows, list(dict.fromkeys(used_cols))


def parse_bool_like(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
        .astype(int)
    )


def add_initial_priority_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], bool]:
    rows = df.copy()
    priority_available = (
        "initial_priority_clean" in rows.columns
        or "initial_priority" in rows.columns
    )

    if "initial_priority_clean" not in rows.columns:
        if "initial_priority" in rows.columns:
            rows["initial_priority_clean"] = (
                rows["initial_priority"]
                .fillna("UNKNOWN")
                .astype(str)
                .str.strip()
                .str.upper()
                .replace({"": "UNKNOWN"})
            )
        else:
            rows["initial_priority_clean"] = "UNKNOWN"
            rows["initial_priority_valid_signal"] = False
            rows["initial_priority_is_set"] = False
            rows["initial_priority_signal_category"] = (
                "NO_PRIORITY_AVAILABLE"
            )
    else:
        rows["initial_priority_clean"] = (
            rows["initial_priority_clean"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"": "UNKNOWN"})
        )

    if "initial_priority_valid_signal" in rows.columns:
        rows["initial_priority_valid_signal_num"] = parse_bool_like(
            rows["initial_priority_valid_signal"]
        )
    else:
        rows["initial_priority_valid_signal_num"] = (
            rows["initial_priority_clean"].isin(["P1", "P2"]).astype(int)
        )

    if "initial_priority_is_set" in rows.columns:
        rows["initial_priority_is_set_num"] = parse_bool_like(
            rows["initial_priority_is_set"]
        )
    else:
        rows["initial_priority_is_set_num"] = (
            ~rows["initial_priority_clean"].isin(["--", "UNKNOWN"])
        ).astype(int)

    if "initial_priority_signal_category" not in rows.columns:
        rows["initial_priority_signal_category"] = np.where(
            rows["initial_priority_valid_signal_num"] == 1,
            "VALID_INITIAL_P1_P2",
            np.where(
                rows["initial_priority_is_set_num"] == 1,
                "INITIAL_LOWER_PRIORITY",
                "EXCLUDE_INITIAL_UNSET",
            ),
        )
        if not priority_available:
            rows["initial_priority_signal_category"] = (
                "NO_PRIORITY_AVAILABLE"
            )

    rows["no_initial_priority_signal"] = (
        rows["initial_priority_valid_signal_num"] == 0
    ).astype(int)

    rank_map = {
        "P1": 1,
        "P2": 2,
        "P3": 3,
        "P4": 4,
        "P5": 5,
        "--": 99,
        "UNKNOWN": 99,
    }
    if "initial_priority_rank" in rows.columns:
        rows["initial_priority_rank_model"] = pd.to_numeric(
            rows["initial_priority_rank"],
            errors="coerce",
        )
    else:
        rows["initial_priority_rank_model"] = (
            rows["initial_priority_clean"].map(rank_map)
        )
    rows["initial_priority_rank_model"] = (
        rows["initial_priority_rank_model"].fillna(99)
    )

    urgency_map = {
        "P1": 5,
        "P2": 4,
        "P3": 3,
        "P4": 2,
        "P5": 1,
        "--": 0,
        "UNKNOWN": 0,
    }
    rows["initial_priority_urgency_score"] = (
        rows["initial_priority_clean"]
        .map(urgency_map)
        .fillna(0)
        .astype(float)
    )

    onehot_cols: list[str] = []
    for value in ["P1", "P2", "P3", "P4", "P5", "--", "UNKNOWN"]:
        suffix = "unset" if value == "--" else value.lower()
        column = f"initial_priority_is_{suffix}"
        rows[column] = (
            rows["initial_priority_clean"] == value
        ).astype(int)
        onehot_cols.append(column)

    initial_cols = [
        "initial_priority_valid_signal_num",
        "no_initial_priority_signal",
        "initial_priority_is_set_num",
        "initial_priority_rank_model",
        "initial_priority_urgency_score",
    ] + onehot_cols

    return rows, initial_cols, priority_available


def add_xgb_blend_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    rows = df.copy()
    text = rows["text_final"].fillna("").astype(str)
    text_lower = text.str.lower()

    rows["text_char_count_xgb"] = text.str.len()
    rows["text_word_count_xgb"] = (
        text.str.split().str.len().fillna(0)
    )
    rows["uppercase_ratio_xgb"] = text.apply(
        lambda value: sum(ch.isupper() for ch in str(value))
        / max(len(str(value)), 1)
    )
    rows["punctuation_density_xgb"] = text.apply(
        lambda value: sum(ch in "!?;:." for ch in str(value))
        / max(len(str(value)), 1)
    )

    rows["summary_len_xgb"] = (
        rows["summary"].fillna("").astype(str).str.len()
        if "summary" in rows.columns
        else 0
    )
    rows["bug_report_text_missing_xgb"] = (
        rows["bug_report_text"].isna().astype(int)
        if "bug_report_text" in rows.columns
        else 0
    )
    rows["text_is_summary_only_xgb"] = (
        (rows["text_word_count_xgb"] <= 40)
        & (rows["text_char_count_xgb"] <= 300)
    ).astype(int)

    signal_cols: list[str] = []
    for column, pattern in XGB_BLEND_SIGNAL_PATTERNS.items():
        rows[column] = text_lower.str.contains(
            pattern,
            regex=True,
        ).astype(int)
        signal_cols.append(column)
    rows["signal_count_xgb"] = rows[signal_cols].sum(axis=1)

    if "initial_priority_clean" in rows.columns:
        priority = (
            rows["initial_priority_clean"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
            .str.strip()
        )
    elif "initial_priority" in rows.columns:
        priority = (
            rows["initial_priority"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
            .str.strip()
        )
    else:
        priority = pd.Series("UNKNOWN", index=rows.index)

    priority = priority.replace(
        {"": "UNKNOWN", "NAN": "UNKNOWN", "NONE": "UNKNOWN"}
    )
    rank_map = {
        "P1": 1,
        "P2": 2,
        "P3": 3,
        "P4": 4,
        "P5": 5,
        "--": 99,
        "UNKNOWN": 99,
    }
    urgency_map = {
        "P1": 5,
        "P2": 4,
        "P3": 3,
        "P4": 2,
        "P5": 1,
        "--": 0,
        "UNKNOWN": 0,
    }

    rows["initial_priority_rank_xgb"] = (
        priority.map(rank_map).fillna(99).astype(float)
    )
    rows["initial_priority_urgency_xgb"] = (
        priority.map(urgency_map).fillna(0).astype(float)
    )
    rows["initial_priority_is_p1_p2_xgb"] = (
        priority.isin(["P1", "P2"]).astype(int)
    )
    rows["no_initial_priority_signal_xgb"] = (
        priority.isin(["--", "UNKNOWN"]).astype(int)
    )
    for value in ["P1", "P2", "P3", "P4", "P5"]:
        rows[f"initial_priority_is_{value.lower()}_xgb"] = (
            priority == value
        ).astype(int)

    extra_numeric_candidates = [
        "comment_count",
        "votes",
        "duplicate_count",
        "depends_count",
        "block_count",
        "keyword_count",
        "flag_count",
        "has_bug_text",
        "has_bug_report_text",
        "has_stack_trace",
        "has_steps_to_reproduce",
        "is_ci_log_like",
        "bug_text_len",
        "model_text_len",
        "summary_length",
    ]
    extra_numeric_cols: list[str] = []
    for column in extra_numeric_candidates:
        if column in rows.columns:
            new_column = f"{column}_xgb_num"
            rows[new_column] = pd.to_numeric(
                rows[column],
                errors="coerce",
            ).fillna(0)
            extra_numeric_cols.append(new_column)

    numeric_cols = [
        "text_char_count_xgb",
        "text_word_count_xgb",
        "uppercase_ratio_xgb",
        "punctuation_density_xgb",
        "summary_len_xgb",
        "bug_report_text_missing_xgb",
        "text_is_summary_only_xgb",
        "signal_count_xgb",
        "initial_priority_rank_xgb",
        "initial_priority_urgency_xgb",
        "initial_priority_is_p1_p2_xgb",
        "no_initial_priority_signal_xgb",
        "initial_priority_is_p1_xgb",
        "initial_priority_is_p2_xgb",
        "initial_priority_is_p3_xgb",
        "initial_priority_is_p4_xgb",
        "initial_priority_is_p5_xgb",
    ] + signal_cols + extra_numeric_cols

    return rows, numeric_cols


def build_xgb_blend_matrix(
    input_df: pd.DataFrame,
    resources: dict[str, Any],
):
    rows, _ = add_xgb_blend_features(input_df)

    numeric_cols = resources["xgb_schema"]["numeric_cols"]
    categorical_cols = resources["xgb_schema"]["categorical_cols"]

    for column in numeric_cols:
        if column not in rows.columns:
            rows[column] = 0

    for column in categorical_cols:
        if column not in rows.columns:
            rows[column] = "UNKNOWN"

    X_text = resources["xgb_blend_tfidf"].transform(
        rows["text_final"]
    )
    X_svd = resources["xgb_blend_svd"].transform(X_text)
    X_numeric = resources["xgb_blend_scaler"].transform(
        rows[numeric_cols].fillna(0)
    )
    X_categorical = resources["xgb_blend_ohe"].transform(
        rows[categorical_cols].fillna("UNKNOWN").astype(str)
    )

    matrix = hstack(
        [
            csr_matrix(X_svd),
            csr_matrix(X_numeric),
            X_categorical,
        ]
    ).tocsr()

    expected_features = resources["xgb_blend_model"].n_features_in_
    if matrix.shape[1] != expected_features:
        raise ValueError(
            "XGBoost blend feature mismatch: "
            f"created {matrix.shape[1]}, expected {expected_features}."
        )

    return matrix


def build_v4_matrix(
    input_df: pd.DataFrame,
    resources: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    rows, _ = add_graphlens_features(input_df)
    rows, _ = add_process_features(rows)
    rows, _, _ = add_initial_priority_features(rows)

    feature_cols = resources["v4_schema"]["feature_cols"]
    for column in feature_cols:
        if column not in rows.columns:
            rows[column] = 0

    metadata = (
        rows[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    X_embedding = resources["encoder"].encode(
        rows["text_final"].tolist(),
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)

    X_tfidf = resources["tfidf"].transform(rows["text_final"])
    X_svd = resources["svd"].transform(X_tfidf)
    X_metadata = resources["scaler"].transform(
        metadata.to_numpy()
    )

    matrix = np.hstack(
        [X_embedding, X_svd, X_metadata]
    ).astype(np.float32)

    expected_features = resources["xgb_main"].n_features_in_
    if matrix.shape[1] != expected_features:
        raise ValueError(
            "V4 feature mismatch: "
            f"created {matrix.shape[1]}, expected {expected_features}."
        )

    return matrix, rows


def _predict_v4_probability(
    matrix: np.ndarray,
    prepared_rows: pd.DataFrame,
    resources: dict[str, Any],
) -> float:
    raw_probability = (
        MAIN_XGB_WEIGHT
        * resources["xgb_main"].predict_proba(matrix)[:, 1]
        + MAIN_LGBM_WEIGHT
        * resources["lgbm_main"].predict_proba(matrix)[:, 1]
    )

    main_probability = resources["calibrator"].predict_proba(
        raw_probability.reshape(-1, 1)
    )[:, 1]

    no_signal = bool(
        int(prepared_rows.iloc[0]["no_initial_priority_signal"])
    )
    xgb_no_signal = resources.get("xgb_no_signal")
    lgbm_no_signal = resources.get("lgbm_no_signal")

    if no_signal and xgb_no_signal is not None and lgbm_no_signal is not None:
        sub_probability = (
            MAIN_XGB_WEIGHT
            * xgb_no_signal.predict_proba(matrix)[:, 1]
            + MAIN_LGBM_WEIGHT
            * lgbm_no_signal.predict_proba(matrix)[:, 1]
        )

        category = str(
            prepared_rows.iloc[0]["initial_priority_signal_category"]
        )
        weight = CATEGORY_SUBMODEL_WEIGHT.get(
            category,
            DEFAULT_SUBMODEL_WEIGHT,
        )

        main_probability[0] = (
            (1.0 - weight) * main_probability[0]
            + weight * sub_probability[0]
        )

    return float(main_probability[0])



def _humanise_signal_name(name: str) -> str:
    cleaned = name

    for prefix in [
        "sig_",
        "risk_",
        "anti_",
        "low_",
        "med_",
    ]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]

    return cleaned.replace("_", " ").strip().title()


def get_stage1_active_signals(
    prepared_rows: pd.DataFrame,
) -> list[str]:
    row = prepared_rows.iloc[0]

    active_signals = []

    for column in GRAPH_PATTERNS:
        if column in prepared_rows.columns:
            if int(row[column]) == 1:
                active_signals.append(
                    _humanise_signal_name(column)
                )

    return active_signals




def predict_stage1(
    summary: str,
    bug_report_text: str = "",
    component: str = "UNKNOWN",
    bug_type: str = "UNKNOWN",
    platform: str = "UNKNOWN",
    op_sys: str = "UNKNOWN",
    initial_priority: str = "UNKNOWN",
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if resources is None:
        resources = load_stage1_resources()

    input_row = build_input_row(
        summary=summary,
        bug_report_text=bug_report_text,
        component=component,
        bug_type=bug_type,
        platform=platform,
        op_sys=op_sys,
        initial_priority=initial_priority,
    )

    v4_matrix, prepared_rows = build_v4_matrix(
        input_row,
        resources,
    )
    v4_probability = _predict_v4_probability(
        v4_matrix,
        prepared_rows,
        resources,
    )

    xgb_matrix = build_xgb_blend_matrix(
        input_row,
        resources,
    )
    xgb_probability = float(
        resources["xgb_blend_model"]
        .predict_proba(xgb_matrix)[0, 1]
    )

    probability_high = (
        BLEND_WEIGHT_V4 * v4_probability
        + BLEND_WEIGHT_XGB * xgb_probability
    )
    is_high = probability_high >= DECISION_THRESHOLD
    label = "HIGH" if is_high else "NOT_HIGH"
    confidence = (
        probability_high if is_high else 1.0 - probability_high
    )

    
    active_signals = get_stage1_active_signals(
        prepared_rows
    )

    stage1_contributions = {
        "V4 semantic ensemble": float(
            BLEND_WEIGHT_V4 * v4_probability
        ),
        "XGBoost baseline": float(
            BLEND_WEIGHT_XGB * xgb_probability
        ),
    }




    return {
        "label": label,
        "probability_high": float(probability_high),
        "confidence": float(confidence),
        "v4_probability_high": float(v4_probability),
        "xgb_probability_high": float(xgb_probability),
        "threshold": DECISION_THRESHOLD,
        "blend_weight_v4": BLEND_WEIGHT_V4,
        "blend_weight_xgb": BLEND_WEIGHT_XGB,
        "priority_category": str(
            prepared_rows.iloc[0][
                "initial_priority_signal_category"
            ]
        ),
        "active_signals": active_signals, 
        "model_contributions": stage1_contributions,
    }
