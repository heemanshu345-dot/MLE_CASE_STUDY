# 🏥 Hospital Readmission Prediction

## Case Study 1 — 30-Day Readmission Risk

A machine learning project that uses **Logistic Regression with L2 regularization** to predict whether a diabetic patient encounter will be followed by a hospital readmission within 30 days.

## 🎯 Objective

Predict the probability of **30-day hospital readmission** from patient demographics, encounter information, previous healthcare utilization, diagnoses, laboratory results, and medication information.

### Target Definition

The original `readmitted` variable contains:

- `NO` → no readmission
- `>30` → readmission after 30 days
- `<30` → readmission within 30 days

For this case study:

```text
<30  → 1 (positive class)
NO   → 0
>30  → 0
```

## 📊 Dataset

The supplied dataset contains **101,766 encounters and 50 columns**.

Important feature groups include:

- Age, gender and race
- Length of hospital stay
- Laboratory procedures
- Procedures
- Number of medications
- Previous outpatient visits
- Previous emergency visits
- Previous inpatient visits
- Primary/secondary/tertiary diagnosis codes
- A1C and glucose serum results
- Insulin and diabetes medications
- Medication changes
- Diabetes medication indicator

`IDS_mapping.csv` is included as a reference file for the dataset's coded IDs.

## 🧠 Methodology

### 1. Data Cleaning

- Replace `?` with missing values
- Create binary target from `readmitted`
- Remove encounter and patient identifiers from model features

### 2. Train/Test Split

A **patient-level GroupShuffleSplit** is used. `patient_nbr` is used only as the grouping variable, so encounters from the same patient do not appear in both training and testing data.

### 3. Preprocessing

Numerical features:

- Median imputation
- StandardScaler

Categorical features:

- Most-frequent imputation
- One-hot encoding
- Unknown categories handled safely

### 4. Model

```python
LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="liblinear",
    max_iter=1000
)
```

L2 regularization reduces the tendency of the model to assign excessively large coefficients to individual features.

## 📈 Evaluation

The primary evaluation metric is **ROC-AUC** because the task is probabilistic binary classification and the positive class is relatively uncommon.

The project also calculates:

- Average Precision
- Precision
- Recall / Sensitivity
- Specificity
- F1-score
- TP / FP / TN / FN

The exact metrics generated from the included dataset are saved to:

```text
results/metrics.csv
```

## ⚕️ Clinical Cost of Errors

### False Negative

A false negative means the model predicts low risk but the patient is actually readmitted within 30 days.

Potential consequences include missed follow-up opportunities, inadequate post-discharge monitoring, and potentially avoidable healthcare utilization.

### False Positive

A false positive means the model flags a patient as high risk but the patient is not readmitted within 30 days.

Potential consequences include additional monitoring, care-coordination workload, and use of healthcare resources.

### Threshold Trade-off

A default probability threshold of `0.50` is not necessarily appropriate for a clinical screening workflow. Lower thresholds generally increase sensitivity while also increasing false positives.

The repository includes an example threshold analysis at several values. The final operational threshold should be chosen using clinical validation, intervention capacity, and quantified costs of false negatives versus false positives.

## 📁 Repository Structure

```text
hospital-readmission-prediction/
│
├── diabetic_data.csv
├── IDS_mapping.csv
├── hospital_readmission_prediction.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── results/
    ├── metrics.csv
    ├── threshold_analysis.csv
    ├── model_coefficients.csv
    ├── roc_curve.png
    ├── precision_recall_curve.png
    └── confusion_matrix_threshold_020.png
```

## 🚀 Installation

```bash
git clone https://github.com/YOUR_USERNAME/hospital-readmission-prediction.git
cd hospital-readmission-prediction
pip install -r requirements.txt
```

## ▶️ Run

```bash
python hospital_readmission_prediction.py
```

After execution, evaluation results and plots are written to the `results/` directory.

## 🔬 Example Output

The script reports:

```text
ROC-AUC: <generated from the supplied dataset>
Average Precision: <generated from the supplied dataset>
```

The repository intentionally generates these values at runtime rather than hard-coding them.

## ⚠️ Limitations

- This is a case-study model, not a clinically validated prediction system.
- The dataset represents diabetic hospital encounters and may not generalize to other populations.
- Diagnosis-code representation is relatively simple compared with modern clinical NLP/embedding approaches.
- External validation is required before deployment.
- Predictions should support, not replace, clinical judgment.

## 🚀 Future Improvements

- XGBoost / Gradient Boosting
- Random Forest
- Class-weighted Logistic Regression
- Hyperparameter tuning
- Diagnosis-code grouping using `IDS_mapping.csv`
- SHAP explainability
- Probability calibration
- Precision-Recall optimization
- Cost-sensitive threshold selection
- External validation

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## 👨‍💻 Author

**Lucky Kashyap**  
Computer Science and Information Technology  
KIET Group of Institutions
