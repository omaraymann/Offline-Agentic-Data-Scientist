# CE888 Offline Agentic Data Scientist

**Student:** Omar Ayman Ebrahem Ebrahem Khalel  
**Module:** CE888 – Data Science and Decision Making  
**Academic Year:** 2025/2026  

---

## 1. Project Overview

This project implements an **offline Agentic Data Scientist** for tabular **classification** datasets.  
The system is designed to run end-to-end on unseen CSV datasets without using LLMs, cloud services, AutoML, or paid APIs.

The agent performs the following stages automatically:

1. load a dataset
2. infer or validate the target column
3. profile dataset characteristics
4. retrieve relevant prior memory
5. generate a data-aware execution plan
6. build preprocessing and model candidates
7. train and evaluate models
8. reflect on the run outcome
9. optionally re-plan and try again
10. save outputs and update persistent memory

The focus of the system is **autonomy, reasoning, adaptability, and explainability**, not just raw predictive accuracy.

---

## 2. System Architecture

The system is organised into modular components:

### Core files

- `run_agent.py`  
  Entry point for running the system from the command line.

- `agentic_data_scientist.py`  
  Main executor class that coordinates the full pipeline.

### Agents

- `agents/planner.py`  
  Generates a plan based on dataset properties such as dataset size, missingness, imbalance, feature types, high cardinality, and weak-signal risk.

- `agents/reflector.py`  
  Analyses model performance after training, diagnoses likely causes of weak performance, and decides whether re-planning is worthwhile.

- `agents/memory.py`  
  Stores prior runs in JSON format and supports both exact fingerprint retrieval and approximate similarity-based hints.

### Tools

- `tools/data_profiler.py`  
  Produces explainable dataset signals such as:
  - feature types
  - missingness severity
  - imbalance ratio
  - number of classes
  - high-cardinality features
  - high-dimensional risk
  - weak-signal risk

- `tools/modelling.py`  
  Builds preprocessing pipelines, selects model candidates, trains models, performs CV where appropriate, and supports decision-threshold tuning for binary imbalance cases.

- `tools/evaluation.py`  
  Produces the final metrics payload, confusion matrix, classification report, and markdown run report.

---

## 3. How Planning Works

The planner converts dataset signals into a structured plan.

Examples of planning decisions:
- if missingness is medium/high -> audit missingness, impute features, add missing indicators
- if categorical cardinality is high -> control high cardinality / group rare categories
- if the target is imbalanced -> use class weighting and balanced metrics
- if the dataset is tiny/small -> enable stronger cross-validation logic
- if the dataset is high-dimensional -> reduce complexity and consider feature selection
- if weak-signal risk is detected -> compare against a dummy baseline and prepare fallback strategies
- if memory suggests good prior strategies -> prioritise those strategies and models

This means the agent does not use a single fixed pipeline for all datasets.

---

## 4. How Reflection and Re-planning Work

After training, the reflector reviews:
- balanced accuracy
- macro F1
- gap versus dummy baseline
- class-level recall
- train/test gaps
- CV stability
- ranking gap to the runner-up model

It then identifies likely root causes such as:
- imbalance handling issues
- preprocessing mismatch
- weak features
- unstable model rankings
- too little data
- data quality issues

If the reflection suggests that another attempt is likely to help, the system:
- updates strategy flags
- expands or adjusts the plan
- runs another attempt

Examples of re-planning actions:
- enable class weighting
- enable CV-aware comparison
- group rare categories
- expand model diversity
- enable threshold tuning

---

## 5. How Memory Works

Persistent memory is stored in `agent_memory.json`.

The system uses:
- **exact memory lookup** using a dataset fingerprint
- **approximate memory lookup** using profile similarity

Memory stores information such as:
- best model on similar datasets
- successful strategies
- failed strategies
- recent run summaries
- common root causes

This allows the planner to reuse prior experience instead of starting from scratch every time.

---

## 6. Models Used

The model pool is selected conditionally based on dataset characteristics and may include:

- DummyClassifier
- LogisticRegression
- RandomForestClassifier
- ExtraTreesClassifier
- GradientBoostingClassifier
- LinearSVC
- KNeighborsClassifier
- SVC

The exact pool depends on:
- dataset size
- numeric vs mixed features
- imbalance
- dimensionality
- high-cardinality categorical columns
- planner and memory flags

---

## 7. Evaluation Strategy

The system evaluates candidate models using metrics suitable for classification tasks.

Metrics include:
- accuracy
- balanced accuracy
- macro F1
- macro precision
- macro recall
- train balanced accuracy
- cross-validation balanced accuracy
- confusion matrix
- classification report

For imbalanced datasets, the system prefers **balanced accuracy** over raw accuracy.

---

## 8. Outputs Produced Automatically

Each run creates a timestamped folder inside `outputs/`.

Typical artefacts include:
- `eda_summary.json`
- `plan.json`
- `metrics.json`
- `reflection.json`
- `selection_rationale.json`
- `attempt_history.json`
- `execution_log.json`
- `confusion_matrix.png`
- `report.md`

These outputs make the system behaviour transparent and reproducible.

---

## 9. How to Run

### Basic command

```bash
python run_agent.py --data data/demo.csv --target auto