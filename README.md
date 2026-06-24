# EXPLAINABLE CROSS-PROJECT BUG SEVERITY PREDICTION USING HYBRID FEATURE FUSION AND TRANSFER LEARNING


**Name:** Syed Izhar Ali  
**Student ID:** 20251328  
**Date:** March 3, 2026  
**Email:** 20251328@mywhitecliffe.com  



Boilerplate repository for a research-oriented bug severity prediction platform that combines:

- classical ML baselines
- BugBERT / transformer text models
- GNN-based cross-project knowledge transfer over InferBugs + BugsRepo
- MLflow experiment tracking and model registry
- FastAPI model serving
- Streamlit dashboards for prediction, monitoring, explainability, and trends

## Core goals

1. Build a reusable ML research repo for bug severity prediction.
2. Track experiments, metrics, params, artifacts, and model versions in MLflow.
3. Keep datasets and pipeline outputs organized for reproducibility.
4. Support multiple modeling tracks: baseline ML, BugBERT, and GNN.
5. Publish model outputs through REST APIs and dashboards.

## Recommended stack

- Python 3.11+
- MLflow
- Streamlit
- FastAPI
- scikit-learn / xgboost
- PyTorch / PyTorch Geometric
- Hugging Face Transformers
- DVC for dataset versioning
- GitHub Actions for CI/CD

## Repository layout

See `docs/PROJECT_BLUEPRINT.md` and `docs/TASK_BREAKDOWN.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# run MLflow locally
mlflow ui --backend-store-uri ./mlruns --port 5000

# run the training pipeline
python pipelines/train_baseline.py

# run API
uvicorn services.api.app:app --reload --port 8000

# run dashboard
streamlit run services/streamlit/Home.py
```

## Suggested execution order

1. Prepare repo and folder structure.
2. Add BugsRepo / InferBugs ingestion scripts.
3. Add baseline TF-IDF + metadata model.
4. Connect baseline runs to MLflow.
5. Add FastAPI serving.
6. Add Streamlit dashboard.
7. Add BugBERT experiments.
8. Add GNN graph build + cache pipeline.
9. Add model registry and deployment workflow.

## Dataset Requirements

This project uses external datasets which are not included in the repository due to size constraints.

### 1. BugsRepo Dataset
Download from:
https://zenodo.org/records/15004067

After downloading, extract and place the files in:
data/raw/BugsRepo/

- Usage:
  - Baseline models
  - Cross-project evaluation

Required files:
- Bug_meta_data_230k_all.csv
- CSV_Contribution_information_dataset.csv
- comments_Dataset_Part_1.csv
- comments_Dataset_Part_2.csv
- comments_Dataset_Part_3.csv
- CSV_100k_filtered_bug_reports.csv
- CSV_Bug_reports_with_steps_to_reproduce.csv

---

### 2. InferredBugs Dataset (for transfer learning / benchmarking)
Download from:
https://github.com/microsoft/InferredBugs

Place the dataset in:
data/raw/InferredBugs/

- Status:
  - Evaluated for transfer learning
  - Found to have significant semantic mismatch with BugsRepo

### 3. Public JIRA Dataset (Under Evaluation)

Download from:
https://zenodo.org/records/5901804

- Source: Zenodo Public JIRA Dataset
- Scale:
  - ~2.7M issues across 16 repositories
- Data:
  - Summary, description, comments
  - Priority (used as proxy for severity)
  - Process features (votes, watchers)

- Purpose:
  - Improve cross-project generalization
  - Reduce semantic gap between datasets

- Current Work:
  - Dataset extraction and preprocessing
  - Vocabulary similarity analysis
  - Transfer learning experiments (planned)


## Project Structure

## After downloading the dataset move it into the respective folders unzipped.
data/
├── raw/
│   ├── BugsRepo/
│   └── InferredBugs/
|   └── ThePublicJiraDataset/
├── processed/


---

### Notes
- The project will not run without placing datasets in the correct directories.
- Data paths are automatically resolved inside the code.


## Notes

- Keep raw data immutable inside `data/raw/`.
- Save intermediate graph caches under `data/interim/graph_cache/`.
- Save final trained models to MLflow and optionally export copies under `models/`.
- Save plots and evaluation reports both to `outputs/` and MLflow artifacts.
