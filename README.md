# Loan-Approval-Prediction

This project predicts **loan approval (Approved/Rejected)** using machine learning. It applies **ensemble models (Stacking Classifier)** with hyperparameter tuning and model explainability using **LIME**.

---

## 🚀 Key Features
- **End-to-end ML pipeline:** EDA → Preprocessing → Model Training → Evaluation.
- **GridSearchCV** for hyperparameter tuning of **Random Forest, Extra Trees, and Logistic Regression** with **Logistic Regression** as metalearner.
- **Stacking Classifier** for improved accuracy.
- **LIME** for model interpretability.
- Achieved **98.2% accuracy** on the test set.

---

## 📊 Dataset
- **Rows:** 4269  
- **Target Variable:** `loan_status` (Approved/Rejected)  
- **Important Features:**  
  `income_annum`, `loan_amount`, `cibil_score`, `loan_term`, `no_of_dependents`, `asset values`, etc.

---

## 🛠 Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, LIME  
- **Environment:** Jupyter Notebook  

---

## 📌 Modeling Approach
1. **Data Cleaning & Label Encoding** – Categorical to numeric conversion.
2. **Exploratory Data Analysis (EDA)** – Visualizations (histograms, correlation heatmaps).
3. **Feature Scaling** – StandardScaler for numerical features.
4. **Model Training** – Random Forest, Extra Trees, Logistic Regression.
5. **Ensemble (Stacking)** – Combines base models with Logistic Regression as meta-learner.
6. **Evaluation** – Accuracy, Confusion Matrix, and Classification Report.
7. **Explainability** – LIME shows feature contributions for individual predictions.

---

## 📈 Performance
- **Accuracy:** 98.2%
- **Confusion Matrix:** 527 TN, 312 TP, 15 misclassifications.
- **Top Feature Impact:** `cibil_score` is the most influential feature.

---

## ⚙️ Requirements
```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
lime
