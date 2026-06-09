=== END FILE ===

=== FILE: data/README.md ===
```markdown
# Data README

This project is designed around three diverse classification datasets so the agent can demonstrate adaptive behaviour across different data conditions.

## 1. Breast Cancer Wisconsin Diagnostic

**Expected file:** `data/breast_cancer.csv`  
**Target column:** `diagnosis`

### Why this dataset is useful
- relatively small and easy to run quickly
- mostly numeric features
- good for showing simpler planning logic and cleaner preprocessing
- useful for demonstrating when linear or distance-based models can compete with ensembles

### What the agent should notice
- low categorical complexity
- likely low missingness
- binary classification
- suitable for threshold tuning if imbalance is present

## 2. Bank Marketing

**Expected file:** `data/bank_marketing.csv`  
**Target column:** `y`

### Why this dataset is useful
- medium-to-large size
- mixed numeric and categorical columns
- realistic tabular business classification problem
- often class-imbalanced
- useful for demonstrating adaptive planning around imbalance and mixed data types

### What the agent should notice
- mixed feature types
- categorical handling matters
- balanced accuracy and macro F1 are more informative than raw accuracy when the target is imbalanced
- runtime-aware model selection matters more than on tiny datasets

## 3. Adult / Census Income

**Expected file:** `data/adult_income.csv`  
**Target column:** `income`

### Why this dataset is useful
- mixed numeric and categorical features
- different structure from Bank Marketing, but still tabular and realistic
- useful for showing how the planner and memory generalise across similar mixed-type datasets
- often contains missing values encoded as strings or blanks depending on the CSV version

### What the agent should notice
- mixed data types
- possible missingness and categorical complexity
- moderate dataset size
- strong candidate for memory-guided planning after other runs

## Recommended preparation notes

- keep all three files in CSV format
- make sure the target columns match the names above
- keep headers intact
- avoid changing class labels unless you also update the target handling consistently

## Demo advice

These three datasets are good together because they expose different branches of the agent:
- Breast Cancer -> mostly numeric, smaller, cleaner
- Bank Marketing -> larger, mixed, potentially imbalanced
- Adult Income -> mixed, realistic missingness and categorical complexity