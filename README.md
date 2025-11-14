# 🩺 Heart Failure Prediction — Machine Learning Project

## 📖 Overview
This project aims to predict the likelihood of **heart disease** using a dataset of patient health indicators such as age, cholesterol, blood pressure, and more.  
The goal is to apply data preprocessing, outlier handling, feature transformation, and multiple machine learning models to find the best predictive performance.

---

## 📂 Dataset
The dataset (`heart.csv`) contains various medical attributes and a binary target variable:
- `HeartDisease`: 1 = presence of heart disease, 0 = no heart disease  
- Numerical features: Age, Cholesterol, RestingBP, MaxHR, Oldpeak, etc.  
- Categorical features: Sex, ChestPainType, ST_Slope, etc.

---

## ⚙️ Steps Performed

### 1. Exploratory Data Analysis (EDA)
- Checked missing values, data types, and distribution of target variable.  
- Visualized numerical and categorical features using Seaborn histograms, countplots, and boxplots.  
- Generated a correlation heatmap for feature relationships.

### 2. Outlier Detection & Winsorization
- Applied an IQR-based method to summarize outliers per numeric column.
- Capped (winsorized) extreme values beyond IQR limits for selected high-outlier features.  
  This helped reduce the influence of extreme values while preserving all data points.

### 3. Preprocessing
- **Numerical pipeline**: Median imputation + RobustScaler  
- **Categorical pipeline**: Mode imputation + OneHotEncoder  
- Combined via `ColumnTransformer` for end-to-end data preparation.

### 4. Baseline Model: Logistic Regression
- Trained baseline Logistic Regression with `class_weight='balanced'` to address class imbalance.  
- Evaluated using ROC-AUC, precision, recall, F1, and confusion matrix.

### 5. Model Improvement
#### a) Logistic Regression (after Winsorization)
- Slightly improved stability and reduced the effect of outliers.

#### b) Logistic Regression (RandomizedSearchCV)
- Tuned hyperparameters (`C`, `solver`) via 5-fold cross-validation.  
- Achieved the best ROC-AUC on validation folds.

#### c) Random Forest Classifier
- Introduced non-linear modeling and feature interaction handling.  
- Generally achieved higher recall and ROC-AUC scores.

#### d) XGBoost Classifier
- Used tree boosting for high-performance gradient optimization.  
- Typically produced the best or near-best ROC-AUC among all models.

---

## 📊 Results Summary (example)
| Model                              | ROC-AUC | Notes |
|------------------------------------|----------|-------|
| Logistic Regression (original)     | 0.82     | Baseline balanced model |
| Logistic Regression (winsorized)   | 0.83     | Reduced outlier effect |
| Logistic Regression (tuned, CV)    | 0.85     | Optimized via RandomizedSearchCV |
| Random Forest                      | 0.87     | Non-linear model, higher recall |
| XGBoost                            | 0.88     | Best performance overall |

---

## 📈 Evaluation Metrics
All models were evaluated on the same test split using:
- **ROC AUC**
- **Precision, Recall, F1-score**
- **Confusion Matrix**
- **ROC Curve**

---

## 💡 Future Work
- Feature importance analysis and SHAP interpretability.
- Add cross-validation on multiple random seeds for stability.
- Try other models (CatBoost, LightGBM, SVM).
- Perform feature selection using Recursive Feature Elimination (RFE).
- Deploy the best model as an API or web dashboard.

---

## 🧰 Tech Stack
- Python (pandas, NumPy, scikit-learn, seaborn, matplotlib)
- XGBoost
- Jupyter Notebook

---

## 👤 Author
**Lazar Šečković**  
Data Science Enthusiast | Machine Learning Learner | Python Developer  
