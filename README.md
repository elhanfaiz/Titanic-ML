# Titanic Survival Prediction – Machine Learning Project

##  Project Overview
This project builds a Machine Learning model to predict whether a passenger survived the Titanic disaster based on passenger attributes such as age, gender, ticket class, fare, and family size.

The goal is to demonstrate a complete end-to-end ML pipeline including data preprocessing, feature engineering, model training, evaluation, and deployment using Streamlit.

---

##  Dataset Information
- Source: Kaggle Titanic Dataset
- Target Variable: `Survived` (0 = No, 1 = Yes)

### Features Used:
- Passenger Class (Pclass)
- Sex
- Age
- SibSp (Siblings/Spouses)
- Parch (Parents/Children)
- Fare
- Embarked
- Engineered Features:
  - FamilySize
  - IsAlone

---

## Data Preprocessing
- Handled missing values (Age, Embarked)
- Dropped irrelevant columns (Name, Ticket, Cabin, PassengerId)
- Converted categorical variables into numerical format
- Feature engineering to improve model performance

---

##  Exploratory Data Analysis (EDA)
- Survival rate by gender
- Survival rate by passenger class
- Age distribution analysis
- Correlation heatmap
- Feature relationships visualization

---

## Machine Learning Models
- Logistic Regression
- Random Forest Classifier

---

## Model Performance

| Model | Accuracy |
|------|----------|
| Logistic Regression | ~0.80 |
| Random Forest | ~0.83 |

✔ Random Forest performed better due to handling non-linear relationships effectively.

---

##  Feature Importance
Most important features influencing survival:
1. Sex (most important)
2. Fare
3. Pclass
4. Age

---

##  Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Feature Importance Analysis

---

## Key Insights
- Women had higher survival probability
- First-class passengers had better survival chances
- Social class strongly influenced survival outcome
- Survival was not random; it depended on socio-economic factors

---

##  Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit (for deployment)

---

##  Deployment
This project is deployed as a web application using Streamlit.
https://jvh43nlntpd8erpsvetytw.streamlit.app/
Live App: (add your link here)

---

##  Future Improvements
- Hyperparameter tuning (GridSearchCV)
- Advanced models (XGBoost, LightGBM)
- Hyperparameter optimization
- Improved UI/UX for web app
- Multi-page ML dashboard

---

##  Author
Machine Learning Project built for portfolio development and real-world deployment practice.

---

