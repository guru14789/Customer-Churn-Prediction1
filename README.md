# Customer-Churn-Prediction1
# Customer Churn Prediction with Machine Learning

This notebook is developed as part of Task 3 during my internship at Codsoft as a Machine Learning Intern. - SREEKUMAR S

## Task
Develop a model to predict customer churn for a subscription-based service or business. Use historical customer data, including features like usage behavior and customer demographics, and apply algorithms like Logistic Regression, Random Forests, or Gradient Boosting to predict churn.

---

## 1. Import Necessary Libraries

```python
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical calculations
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler  # For standardizing features
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Ensemble models
from sklearn.linear_model import LogisticRegression  # Linear model for binary classification
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score  # For model evaluation
from sklearn.utils import resample
```

## 2. Import the Dataset
```
data = pd.read_csv('/content/Churn_Modelling.csv')
data.head(10)  # Display first 10 rows for understanding the dataset
```

## 3. Dataset Summary
```
data.info()  # Summary of the DataFrame
```

## 4. Check for Missing Values
```
data.isnull().sum()
```
## 5. Data Preprocessing
```
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)  # Drop irrelevant columns
data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)  # One-hot encoding
```

## 6. Split Features and Target Variables
```
X = data.drop('Exited', axis=1)  # Features (all columns except 'Exited')
y = data['Exited']  # Target variable
```

## 7. Split Data into Train and Test Sets
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## 8. Standardize Features
```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
## 9. Train Logistic Regression Model
```
log = LogisticRegression(random_state=42)
log.fit(X_train, y_train)
y_pred0 = log.predict(X_test)
log_acc = accuracy_score(y_test, y_pred0)
print(f'Logistic Regression Accuracy: {log_acc:.4f}')
print(classification_report(y_test, y_pred0))

```
## 10. Train Random Forest Model
```
random = RandomForestClassifier(n_estimators=100, random_state=42)
random.fit(X_train, y_train)
y_pred1 = random.predict(X_test)
random_acc = accuracy_score(y_test, y_pred1)
print(f'Random Forest Accuracy: {random_acc:.4f}')
print(classification_report(y_test, y_pred1))
```
## 11. Train Gradient Boosting Model
```
gradient = GradientBoostingClassifier(n_estimators=100, random_state=42)
gradient.fit(X_train, y_train)
y_pred2 = gradient.predict(X_test)
gradient_acc = accuracy_score(y_test, y_pred2)
print(f'Gradient Boosting Accuracy: {gradient_acc:.4f}')
print(classification_report(y_test, y_pred2))
```
## 12. Model Evaluation
```
print(f"Logistic Regression ROC AUC: {roc_auc_score(y_test, y_pred0):.4f}")
print(f"Random Forest ROC AUC: {roc_auc_score(y_test, y_pred1):.4f}")
print(f"Gradient Boosting ROC AUC: {roc_auc_score(y_test, y_pred2):.4f}")
```
```
print("Confusion Matrix for Logistic Regression:")
print(confusion_matrix(y_test, y_pred0))
print("Confusion Matrix for Random Forest:")
print(confusion_matrix(y_test, y_pred1))
print("Confusion Matrix for Gradient Boosting:")
print(confusion_matrix(y_test, y_pred2))
```

