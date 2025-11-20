# Soil Fertility Prediction using Machine Learning

---

## Project Description and Purpose

This project develops an intelligent machine learning system for **binary soil fertility classification** (Infertile vs. Fertile) using physicochemical soil properties. The primary goal is to provide farmers and agricultural practitioners with a rapid, cost-effective, and accurate alternative to traditional laboratory-based soil testing methods.

### Key Objectives:
- Classify soil samples into two fertility categories: **Class 0 (Infertile)** and **Class 1 (Fertile)**
- Implement domain-inspired feature engineering based on agronomic principles
- Compare multiple ML algorithms (Random Forest, XGBoost, Voting Ensemble)
- Address class imbalance using advanced resampling techniques (BorderlineSMOTE, Tomek Links)
- Achieve state-of-the-art classification performance with interpretable results

### Real-World Impact:
- **Cost Reduction:** Basic soil test (~$50) vs. comprehensive lab analysis (~$200)
- **Time Efficiency:** Instant predictions vs. 7-10 days lab processing
- **Accessibility:** Enables small-scale farmers in resource-constrained regions to make data-driven decisions

---

## Dataset Information

**Dataset:** `Soil_fertility_data.csv`

| Property | Value |
|----------|-------|
| **Total Samples** | 880 (original) → 841 (after removing Class 2) |
| **Features** | 12 soil physicochemical parameters |
| **Target Variable** | Binary classification (0 = Infertile, 1 = Fertile) |
| **Missing Values** | 0 (clean dataset) |

### Input Features:

| Attribute | Description             | Type        | Range/Values               |
| --------- | ----------------------- | ----------- | -------------------------- |
| N         | Nitrogen                | Numeric     | 6 - 383 kg/ha              |
| P         | Phosphorus              | Numeric     | 2.9 - 125 kg/ha            |
| K         | Potassium               | Numeric     | 11 - 887 kg/ha             |
| pH        | Soil pH                 | Numeric     | 0.9 - 11.15                |
| EC        | Electrical Conductivity | Numeric     | 0.1 - 0.95 dS/m            |
| OC        | Organic Carbon          | Numeric     | 0.1 - 24%                  |
| S         | Sulfur                  | Numeric     | 0.64 - 31 ppm              |
| Zn        | Zinc                    | Numeric     | 0.07 - 42 ppm              |
| Fe        | Iron                    | Numeric     | 0.21 - 44 ppm              |
| Cu        | Copper                  | Numeric     | 0.09 - 3.02 ppm            |
| Mn        | Manganese               | Numeric     | 0.11 - 31 ppm              |
| B         | Boron                   | Numeric     | 0.06 - 2.82 ppm            |
| Output    | Fertility Class         | Categorical | 0 (Infertile), 1 (Fertile) |

-**Total Samples:** 841 (after removing Class 2)
-**Train Set:** 672 samples (80%)
-**Test Set:** 169 samples (20%)

**Preprocessing Note:** Original dataset contained 3 classes (0, 1, 2). Class 2 (highly fertile, 39 samples) was removed to create a balanced binary classification problem.

---

## 🚀 Steps to Execute the Code

### Prerequisites

Ensure you have **Python 3.8+** installed on your system.

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd soil-fertility-prediction
```

### Step 2: Install Required Libraries

Install all dependencies using pip:

```python
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve)                   
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTETomek
import warnings
```

### Step 3: Prepare Dataset

1. Place `Soil_fertility_data.csv` in your working directory or Google Drive
2. Update the file path in the notebook:

```python
df = pd.read_csv('path/to/Soil_fertility_data.csv')
```

### Step 4: Run the Notebook

**Option A: Google Colab (Recommended)**
1. Upload `SOIL_Fert-2.ipynb` to Google Colab
2. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Update dataset path to your Drive location
4. Run all cells sequentially (Runtime → Run all)

### Step 5: View Results

The notebook will output:
- Dataset overview and statistics
- Exploratory data analysis (EDA) visualizations
- Model training progress
- Performance comparison table
- Feature importance charts
- Confusion matrices
- ROC curves
- Cross-validation results

---

## Required Python Libraries and Versions

### Core Libraries:

| Library | Version | Purpose |
|---------|---------|---------|
| **pandas** | ≥1.3.0 | Data manipulation and analysis |
| **numpy** | ≥1.21.0 | Numerical computing |
| **matplotlib** | ≥3.4.0 | Data visualization |
| **seaborn** | ≥0.11.0 | Statistical visualizations |
| **scikit-learn** | ≥1.0.0 | Machine learning algorithms |
| **xgboost** | ≥1.5.0 | Gradient boosting classifier |
| **imbalanced-learn** | ≥0.9.0 | SMOTE and resampling techniques |

### Installation Command:

```bash
pip install pandas>=1.3.0 numpy>=1.21.0 matplotlib>=3.4.0 seaborn>=0.11.0 scikit-learn>=1.0.0 xgboost>=1.5.0 imbalanced-learn>=0.9.0
```

## Input/Output Formats

**Input:**  
CSV format with columns as above, target column `Output` (0/1).

**Output:**  
- Console reports: Accuracy, Precision, Recall, F1-score, Confusion Matrix  
- PNG / PDF images: Feature importance chart, ROC curves, Confusion matrix  
- Model artifacts (`xgboost_model.pkl`, `scaler.pkl`) can be exported for deployment

---

## Detailed Performance Metrics and Configuration

### Table 1: Model Performance Comparison

| Model           | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | ROC-AUC |
| --------------- | ------------- | --------------- | ------------ | -------------- | ------- |
| Random Forest   | 0.9290        | 0.9290          | 0.9290       | 0.9288         | 0.9620  |
| XGBoost (Best)  | 0.9349        | 0.9349          | 0.9348       | 0.9348         | 0.9714  |
| Voting Ensemble | 0.9290        | 0.9290          | 0.9290       | 0.9288         | 0.9620  |

### Table 2: Cross-Validation Results (5-Fold Stratified)

| Model          | CV Accuracy     | CV Macro F1     | CV ROC-AUC     | Stability (Std Dev) |
| -------------- | --------------- | --------------- | -------------- | ------------------- |
| Random Forest  | 0.9441 ± 0.0132 | 0.9440 ± 0.0132 | 0.9625 ± 0.011 | Moderate            |
| XGBoost (Best) | 0.9477 ± 0.0069 | 0.9476 ± 0.0069 | 0.9723 ± 0.007 | High                |

- Lower standard deviation indicates more stable performance across different data splits.

### Table 4: Engineered Features List

| Feature Category     | Feature Name          | Formula/Description               |
| -------------------- | --------------------- | --------------------------------- |
| Nutrient Ratios      | NPK_ratio             | N / (P + K)                       |
|                      | NP_ratio              | N / P                             |
|                      | NK_ratio              | N / K                             |
|                      | PK_ratio              | P / K                             |
| Interaction Terms    | N_pH_interaction      | N × pH                            |
|                      | pH_OC_interaction     | pH × OC                           |
|                      | EC_OC_interaction     | EC × OC                           |
| Composite Indices    | macronutrient_index   | (N + P + K) / 3                   |
|                      | micronutrient_index   | (Zn + Fe + Cu + Mn + B) / 5       |
|                      | total_nutrients       | Sum of all nutrients              |
| Agronomic Principles | limiting_nutrient     | min(N_norm, P_norm, K_norm)       |
|                      | nutrient_balance      | Balance score based on NPK ratios |
| Soil Quality Proxies | soil_quality_proxy    | (OC × pH) / EC                    |
|                      | CEC_proxy             | OC × 10 + EC × 5                  |
|                      | base_saturation_proxy | (pH - 4) / (9 - 4)                |

### Table 5: Hyperparameters Used

| Model         | Key Hyperparameters | Values   |
| ------------- | ------------------- | -------- |
| Random Forest | n_estimators        | 200      |
|               | max_depth           | 15       |
|               | min_samples_split   | 5        |
|               | class_weight        | balanced |
|               | random_state        | 42       |
| XGBoost       | n_estimators        | 200      |
|               | learning_rate       | 0.05     |
|               | max_depth           | 5        |
|               | subsample           | 0.8      |
|               | colsample_bytree    | 0.8      |
|               | random_state        | 42       |

- **Feature Importance:** Engineered domain features (N×pH, NK_ratio, etc.) make up >65% model importance.

---

## Method Overview

- **Data Preprocessing:** Remove Class 2, stratified split, standard scaling  
- **Feature Engineering:** 15 new features—ratios, interactions, indices—added to original 12  
- **Resampling:** BorderlineSMOTE & Tomek Links for class imbalance (if needed)  
- **Model Training:** Random Forest, XGBoost, Voting Ensemble  
- **Evaluation:** 5-fold CV, detailed test metrics, confusion matrix, ROC/AUC  
- **Interpretability:** SHAP/feature importance plots  
- **Artifacts:** Model (.pkl), scaler, feature names, visualizations  

---

## References

1. Gunasekaran, K., et al. (2025). *Frontiers in Soil Science*, 5, 1652058.
2. Nwamekwe, C. O., et al. (2025). *GU J Sci, Part A*, 12(1), 36-60.
3. Chawla, N. V., et al. (2002). *JAIR*, SMOTE.

---

## Contact

- Email: [pravin267135@gmail.com]
- GitHub: [https://github.com/22Pravin/]

---

**Last Updated:** November 20, 2025


