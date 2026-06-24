
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / (
        "bugsrepo_curated_frozen_combined_audit_filtered_"
        "aggressive_with_initial_priority_with_metadata_creator_exp_"
        "PLUS_10K_Firefox_10K_Thunderbird_BALANCED_CLEANED.csv"
    )
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "services"
    / "streamlit"
    / "dashboard_data"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalise_severity(series: pd.Series) -> pd.Series:
    mapping = {
        "blocker": "HIGH",
        "critical": "HIGH",
        "high": "HIGH",
        "s1": "HIGH",
        "s2": "HIGH",
        "major": "MEDIUM",
        "normal": "MEDIUM",
        "medium": "MEDIUM",
        "s3": "MEDIUM",
        "minor": "LOW",
        "trivial": "LOW",
        "low": "LOW",
        "s4": "LOW",
    }

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )


df = pd.read_csv(DATASET_PATH, low_memory=False)

original_column_count = len(df.columns)

project_col = "project" if "project" in df.columns else "product"

df["project_display"] = (
    df[project_col]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

df["severity_three_class"] = normalise_severity(df["severity"])

df["stage1_label"] = (
    df["label_stage1"]
    .fillna("UNKNOWN")
    .astype(str)
    .str.strip()
    .str.upper()
    .replace(
        {
            "NOT-HIGH": "NOT_HIGH",
            "NOT HIGH": "NOT_HIGH",
        }
    )
)

summary = {
    "total_rows": int(len(df)),
    "total_columns": int(original_column_count),
    "project_count": int(df["project_display"].nunique()),
    "high_count": int((df["stage1_label"] == "HIGH").sum()),
    "not_high_count": int(
        (df["stage1_label"] == "NOT_HIGH").sum()
    ),
    "rows_before_final_cleaning": 56730,
    "rows_retained": 52099,
    "rows_removed": 4631,
    "missing_text_removed": 107,
    "ci_log_like_removed": 4524,
}

with open(
    OUTPUT_DIR / "dataset_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=2)


project_counts = (
    df["project_display"]
    .value_counts()
    .rename_axis("Project")
    .reset_index(name="Bug reports")
)
project_counts.to_csv(
    OUTPUT_DIR / "project_counts.csv",
    index=False,
)


severity_counts = (
    df["severity_three_class"]
    .dropna()
    .value_counts()
    .reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
    .rename_axis("Severity")
    .reset_index(name="Bug reports")
)
severity_counts.to_csv(
    OUTPUT_DIR / "severity_counts.csv",
    index=False,
)


project_severity_counts = (
    df[df["severity_three_class"].notna()]
    .groupby(["project_display", "severity_three_class"])
    .size()
    .reset_index(name="Bug reports")
)
project_severity_counts.to_csv(
    OUTPUT_DIR / "project_severity_counts.csv",
    index=False,
)


stage1_counts = (
    df["stage1_label"]
    .value_counts()
    .reindex(["HIGH", "NOT_HIGH"], fill_value=0)
    .rename_axis("Stage-1 class")
    .reset_index(name="Bug reports")
)
stage1_counts.to_csv(
    OUTPUT_DIR / "stage1_counts.csv",
    index=False,
)


metadata_columns = [
    column
    for column in [
        "component",
        "type",
        "platform",
        "op_sys",
        "initial_priority",
        "initial_priority_clean",
        "creator_experience_level",
    ]
    if column in df.columns
]

metadata_rows = []

for column in metadata_columns:
    values = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    missing_count = int(
        values.isin(
            ["", "nan", "none", "unknown", "--"]
        ).sum()
    )

    metadata_rows.append(
        {
            "Feature": column,
            "Available rows": len(df) - missing_count,
            "Missing/unknown rows": missing_count,
            "Completeness (%)": (
                (len(df) - missing_count) / len(df) * 100
            ),
        }
    )

pd.DataFrame(metadata_rows).to_csv(
    OUTPUT_DIR / "metadata_completeness.csv",
    index=False,
)


sample_columns = [
    column
    for column in [
        "project_display",
        "severity",
        "severity_three_class",
        "stage1_label",
        "summary",
        "component",
        "type",
        "platform",
        "op_sys",
        "initial_priority",
    ]
    if column in df.columns
]

sample_df = (
    df[sample_columns]
    .sample(
        n=min(3000, len(df)),
        random_state=42,
    )
    .reset_index(drop=True)
)

sample_df.to_csv(
    OUTPUT_DIR / "dataset_sample.csv",
    index=False,
)

print("Dashboard dataset assets created in:")
print(OUTPUT_DIR)

for path in sorted(OUTPUT_DIR.iterdir()):
    print(f"{path.name}: {path.stat().st_size / 1024:.1f} KB")
