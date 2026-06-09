# Offline Agentic Data Scientist

A fully offline agentic data science system that autonomously performs end-to-end classification on unseen tabular datasets — without any LLMs or external APIs.

## Overview

This project was built for CE888 (Data Science and Decision Making) at the University of Essex. The agent behaves like an offline data scientist: it reasons about the dataset, builds a conditional plan, trains and compares models, reflects on performance, and remembers past runs.

## Architecture

```
run_agent.py                  # CLI entry point
agentic_data_scientist.py     # Orchestrates the run loop
agents/
├── planner.py                # Generates conditional execution plan
├── reflector.py              # Diagnoses performance and triggers replanning
└── memory.py                 # Persistent memory across runs
tools/
├── data_profiler.py          # Extracts dataset signals and strategy flags
├── modelling.py              # Preprocessing, model training, CV ranking
└── evaluation.py             # Metrics, confusion matrix, selection rationale
outputs/                      # Reports, metrics, plots saved here
```

## How to Run

```bash
pip install -r requirements.txt

python run_agent.py \
  --data data/your_dataset.csv \
  --target auto \
  --seed 42 \
  --test_size 0.2 \
  --max_replans 1
```

## Key Features

### 1. Data Profiling
The profiler converts raw columns into interpretable signals: missingness severity, class imbalance ratio, cardinality, dataset size bucket, and weak-signal risk. Every downstream decision is driven by these signals.

### 2. Conditional Planning
The planner expands a base workflow with conditional steps. For example:
- High missingness → impute and add missing indicators
- Class imbalance → activate class weighting and use balanced accuracy
- Tiny dataset → prefer simple models and enable cross-validation

### 3. Model Selection
The agent selects a model pool suited to the dataset profile — not a fixed list. Baselines (Dummy, Logistic Regression, Random Forest) are always included. Gradient Boosting, LinearSVC, KNN, and SVC are conditionally added based on dataset size and risk flags.

### 4. Self-Correcting Reflector
After training, the reflector diagnoses performance by analysing:
- Balanced accuracy and macro F1
- Train-test gap (overfitting signal)
- Worst-class recall
- Cross-validation stability

If performance is weak and a realistic improvement path exists, the reflector updates strategy flags and triggers replanning.

### 5. Persistent Memory
The agent stores dataset fingerprints, successful models, failed strategies, and reflection root causes in `agent_memory.json`. Future runs on similar datasets start with better priors.

## Outputs Per Run

| File | Description |
|------|-------------|
| `eda_summary.json` | Dataset profile and signals |
| `plan.json` | Generated execution plan |
| `metrics.json` | All candidate model metrics |
| `reflection.json` | Reflection diagnosis |
| `selection_rationale.json` | Why the final model was chosen |
| `attempt_history.json` | Full attempt log |
| `confusion_matrix.png` | Confusion matrix of best model |
| `report.md` | Human-readable summary |

## Constraints
- Fully offline — no LLMs, no paid APIs, no AutoML platforms
- Python only — pandas, NumPy, scikit-learn, matplotlib

## Author
Omar Ayman Ebrahem Khalel  
MSc Artificial Intelligence, University of Essex  
