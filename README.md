# credit-pre-screening---Bussiness_UseCase-ML
Credit Pre-Screening Engine using Machine Learning for loan intake optimisation
# 🏦 Credit Pre-Screening Engine — Loan Intake Optimisation

### An End-to-End Machine Learning Project for Banking and Retail Lending

**Domain:** Banking, Financial Services and Insurance (BFSI)  
**Project:** Phase 2 — Mini Use Case 02  
**Author:** Madhu Kunchala  
**GitHub:** [Madhukunchala99](https://github.com/Madhukunchala99)

---

## 📌 Project Overview

The Credit Pre-Screening Engine is a machine learning project designed to reduce the manual workload involved in personal loan application reviews.

The system analyses historical bank decisions and identifies applications suitable for three screening lanes:

- **Auto-Decline:** Applications with very low predicted approval probability.
- **Fast-Track:** Applications with very high predicted approval probability, subject to verification.
- **Manual Review:** Uncertain applications requiring human judgement.

The goal is to improve operational efficiency while keeping observed historical decision errors within a **5% limit for each automated lane**.

## 🎯 Business Problem

A bank processes approximately:

| Business Metric | Value |
|---|---|
| Monthly loan applications | 3,000 |
| Average manual review time | 25 minutes |
| Monthly analyst workload | 1,250 hours |
| Maximum historical error per automated lane | 5% |

Reviewing every application manually consumes significant time and resources.

**Objective:** Identify clear-cut applications that can bypass full manual review, allowing credit analysts to focus on complex cases.

## 📊 Dataset Overview

The project uses a simulated historical loan application dataset.

| Property | Details |
|---|---|
| Total records | 12,000 |
| Original columns | 19 |
| Historical period | April–September 2026 |
| Target variable | `decision` |
| Approved applications | 6,353 (52.94%) |
| Declined applications | 5,647 (47.06%) |

Key features include credit score, monthly income, employment type, existing EMI, proposed EMI, loan amount, loan purpose and previous delinquencies.

**Important:** The target represents historical bank approval decisions, not actual loan repayment or default outcomes.

## 🔍 Project Workflow

### 1. Data Inspection and Cleaning

- Identified 441 missing monthly income values and handled them using employment-wise median imputation.
- Investigated 1,039 missing credit scores associated with applicants having no credit history.
- Created a new-to-credit indicator instead of dropping these applicants.
- Identified 25 suspicious annual-income entries recorded as monthly income and divided them by 12 for the simulated dataset.
- Retained all 12,000 applications.

### 2. Exploratory Data Analysis (EDA)

Analysed historical loan approval patterns across:

- Credit-score bands.
- Employment categories.
- Income levels.
- Repayment obligations.
- Loan affordability.

**Key observations:**

- Credit scores below 650 were associated with very low approval rates.
- Higher credit scores generally corresponded to higher historical approval rates.
- Salaried applicants had higher observed approval rates than gig and contract workers.
- Applicants with higher EMI burdens generally had lower approval rates.

These findings describe associations, not necessarily causal relationships.

### 3. Feature Engineering

Three important features were developed:

**Fixed Obligation to Income Ratio (FOIR)**

`FOIR = (Existing EMI + Proposed EMI) / Monthly Income`

Measures the proportion of monthly income committed to loan repayments.

**Loan-to-Income Ratio**

`Loan-to-Income = Loan Amount / (Monthly Income × 12)`

Measures the requested loan amount relative to annual income.

**New-to-Credit Indicator (`is_ntc`)**

Identifies applicants without an existing credit score.

### 4. Logistic Regression Baseline

Logistic Regression was used as the initial interpretable binary classification model.

- 80:20 stratified train-test split.
- 9,600 training records.
- 2,400 testing records.
- Numerical scaling and categorical encoding.
- Evaluation using accuracy, precision, recall, ROC-AUC and confusion matrix.

**Baseline results:**

| Metric | Result |
|---|---:|
| Accuracy | 78.54% |
| Precision — Declined | 78.58% |
| Recall — Declined | 74.76% |
| ROC-AUC | 0.8671 |

### 5. Threshold Optimisation

The default classification threshold of 0.50 produced too many incorrect decline predictions for automated rejection.

Multiple decline-probability thresholds were evaluated.

A threshold of **0.90** produced:

- **14.62% auto-decline coverage**
- **4.27% historical wrong-decline rate**
- Approximately **183 gross analyst hours saved per month**

This provided an initial screening policy but left most applications for manual review.

### 6. Fairness Analysis

Gender-based fairness checks were conducted to identify potential differences in screening outcomes.

At the 0.90 decline-probability threshold:

| Group | Auto-Decline Rate |
|---|---:|
| Female | 15.33% |
| Male | 14.30% |

The observed difference was **1.03 percentage points**.

Removing gender from model inputs does not guarantee fairness because features such as income and employment type may act as indirect proxies.

These results indicate the need for further fairness assessment and monitoring.

### 7. Reusable Machine Learning Pipeline

A reusable workflow was developed using Scikit-learn's `Pipeline` and `ColumnTransformer`.

The workflow handles:

- Feature creation.
- Missing-value imputation.
- Numerical scaling.
- Categorical encoding.
- Model prediction.

Training-dependent preprocessing is learned from training data to reduce data leakage.

### 8. Model Comparison

Four classification models were compared using **5-fold Stratified Cross-Validation**.

| Model | Mean ROC-AUC |
|---|---:|
| **Gradient Boosting** | **0.8896** |
| Random Forest | 0.8773 |
| Logistic Regression | 0.8667 |
| Decision Tree | 0.7187 |

**Gradient Boosting was selected** because it achieved the highest mean cross-validation ROC-AUC among the evaluated models.

Its test-set performance included:

- Accuracy: **80.29%**
- ROC-AUC: **0.8853**
- Precision — Declined: **81.54%**
- Recall — Declined: **75.11%**

### 9. Three-Lane Credit Screening

The selected Gradient Boosting model was used to design a two-threshold screening policy based on predicted approval probability.

| Screening Lane | Approximate Approval Probability | Action |
|---|---|---|
| Auto-Decline | Below 17% | Automated decline recommendation |
| Manual Review | Between thresholds | Human credit assessment |
| Fast-Track | 89% or above | Expedited verification |

**Observed screening results:**

- Auto-Decline: approximately **24.5%**
- Fast-Track: approximately **19.0%**
- Manual Review: approximately **56.5%**
- Combined automation coverage: approximately **43.5%**

The observed historical disagreement rates were below 5% in both automated lanes.

### 10. K-Means Applicant Segmentation

K-Means clustering was performed as an additional exploratory analysis.

Four applicant segments were identified, including:

- Higher-income applicants.
- Moderate-income applicants with shorter tenure.
- Applicants with established employment.
- Applicants with high FOIR and affordability concerns.

The higher-income segment showed the highest model prediction error, approximately **26.13%**.

Another segment had an average FOIR of approximately **72%**.

Clustering helped identify differences in applicant characteristics and model performance. It was not used to replace the supervised classifier.

## 📈 Business Impact

The selected Gradient Boosting approach achieved approximately **43.5% automation coverage**.

Using the business assumption of 3,000 monthly applications and 25 minutes per full review, this corresponds to:

| Estimated Impact | Value |
|---|---:|
| Applications avoiding full manual review | ~1,305/month |
| Potential gross analyst hours freed | ~544/month |
| Remaining applications for full review | ~56.5% |
| Historical error target | Under 5% per automated lane |

These are estimates of avoided full-review effort, not guaranteed net savings. Fast-track verification and operational controls still require resources.

## 🛠️ Technologies Used

- **Python** — Programming and analysis
- **Pandas and NumPy** — Data processing
- **Matplotlib and Seaborn** — Data visualisation
- **Scikit-learn** — Preprocessing, classification, clustering and evaluation
- **Jupyter Notebook** — Experimentation and documentation

## ▶️ Running the Project

Clone the repository:

```bash
git clone https://github.com/Madhukunchala99/credit-pre-screening---Bussiness_UseCase-ML.git
```

Navigate into the project folder:

```bash
cd credit-pre-screening---Bussiness_UseCase-ML
```

Install the main Python dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook, check the dataset path and run the analysis cells in order.

## ⚠️ Limitations and Future Improvements

- The dataset contains simulated historical decisions rather than real loan repayment outcomes.
- Historical decision agreement does not prove creditworthiness or repayment safety.
- Screening thresholds require independent validation before production use.
- Fairness must be evaluated across relevant applicant groups.
- Data quality, model performance and approval patterns may change over time.
- Production use would require auditability, human oversight and ongoing monitoring.

Future work could include independent time-based validation, probability calibration, more comprehensive fairness testing and deployment monitoring.

## ✅ Conclusion

This project demonstrates how machine learning can support a more efficient personal-loan screening process.

Gradient Boosting achieved stronger discrimination than the baseline models and enabled a three-lane screening approach with approximately **43.5% observed automation coverage**.

The central idea is to **automate clear cases cautiously while keeping uncertain decisions under human review**.

---

**Developed by Madhu Kunchala**  
[GitHub Profile](https://github.com/Madhukunchala99)
