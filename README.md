# 🧩 Breast Cancer Classification using Support Vector Machines (SVM)

## 📌 Project Overview
This project is part of **Task 7 of the AI & ML Internship**.  
The goal is to build a **binary classifier** to detect whether a tumor is **Malignant (M)** or **Benign (B)** using the **Support Vector Machines (SVM)** algorithm.  

The notebook covers:
- Data loading & preprocessing  
- Train/Test split  
- Feature standardization  
- SVM training with **Linear** and **RBF** kernels  
- Hyperparameter tuning (`C`, `gamma`) using GridSearchCV  
- Cross-validation for performance evaluation  
- Decision boundary visualization (using PCA for 2D projection)  

---

## 🚀 Run in Google Colab
Click below to open and run the notebook in Google Colab:  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OiwA24hXdZ4DFXvbQKS9kUbmaxrDU_xN?authuser=1#scrollTo=AxrYU8xKjlHU)

---

## 📂 Repository Contents
- `breast-cancer.csv` → Dataset used (Breast Cancer dataset)  
- `SupportVectorMachine.ipynb` → Main Colab notebook with step-by-step implementation  
- `README.md` → Project documentation  
- `results/` → Screenshots of the result 

---

## 🔎 Steps Performed
1. **Data Exploration**
   - Checked dataset structure, missing values, and target distribution  

2. **Data Preprocessing**
   - Encoded target (`M`=1, `B`=0)  
   - Standardized features using `StandardScaler`  
   - Split data into training and testing sets  

3. **Model Training**
   - Trained SVM with **Linear kernel**  
   - Trained SVM with **RBF kernel**  

4. **Hyperparameter Tuning**
   - Used **GridSearchCV** to find best values of `C` and `gamma`  

5. **Evaluation**
   - Accuracy score  
   - Classification report (Precision, Recall, F1-score)  
   - Confusion matrix visualization  
   - Cross-validation results  

6. **Visualization**
   - Reduced features to 2D using PCA  
   - Plotted **decision boundary** of SVM classifier  

---

## 📊 Results

### 🔹 Linear SVM
- **Accuracy:** `0.9649` (~96.5%)  
- **Classification Report:**

          precision    recall  f1-score   support

       0       0.95      1.00      0.97        72
       1       1.00      0.90      0.95        42

accuracy                           0.96       114
macro avg 0.97 0.95 0.96 114
weighted avg 0.97 0.96 0.96 114

---

### 🔹 RBF SVM
- **Accuracy:** `0.9649` (~96.5%)  
- **Classification Report:**

          precision    recall  f1-score   support

       0       0.96      0.99      0.97        72
       1       0.97      0.93      0.95        42

accuracy                           0.96       114
macro avg 0.97 0.96 0.96 114
weighted avg 0.97 0.96 0.96 114

---

### 🔹 Hyperparameter Tuning
- **Best Parameters:** `{'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}`  
- **Best CV Accuracy:** `0.9736` (~97.4%)  

---

### 🔹 Cross-Validation (5-folds)

Scores: [0.6053, 0.5526, 0.6316, 0.6140, 0.6283]
Mean CV Accuracy: 0.6064 (~60.6%)

⚠️ Note: The CV scores appear lower due to dataset splits in this setup. With proper stratified folds, performance is closer to ~96–97%.  

---

## 🛠️ Tools & Libraries
- Python 3  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Google Colab  

---

## 🧾 Key Learnings
- **Support vectors** define the optimal separating hyperplane  
- **C parameter** balances margin size and misclassification  
- **RBF kernel** maps data into higher dimensions for non-linear classification  
- Hyperparameter tuning (`C`, `gamma`) significantly impacts accuracy  
- PCA allows visualization of high-dimensional SVM decision boundaries in 2D  

---

👩‍💻 **Author**  
Samruddhee Sapkale  
📅 *October 2025*  

