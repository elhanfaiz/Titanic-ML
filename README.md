#  Titanic Survival Prediction using Machine Learning

## Project Overview
This project predicts whether a passenger survived the Titanic disaster using machine learning models. The dataset is from Kaggle and includes passenger details like age, sex, class, fare, etc.

---

##  Dataset
- Source: Kaggle Titanic Dataset  
- Features: Passenger information (Sex, Age, Fare, Pclass, etc.)  
- Target: Survived (0 = No, 1 = Yes)

---

## Data Preprocessing
- Handled missing values (Age, Embarked)
- Dropped unnecessary columns (Name, Ticket, Cabin, PassengerId)
- Encoded categorical variables (Sex, Embarked)
- Created new features:
  - FamilySize = SibSp + Parch
  - IsAlone

---

##  Exploratory Data Analysis
- Survival rate by gender
- Survival rate by passenger class
- Age distribution analysis
- Correlation heatmap

---

## Machine Learning Models
- Logistic Regression
- Random Forest Classifier

---

##  Model Performance

| Model | Accuracy |
|------|----------|
| Logistic Regression | ~0.80 |
| Random Forest | ~0.83 |

 Random Forest performed better due to non-linear relationships in data.

---

## Feature Importance
Top important features:
- Sex (most important)
- Fare
- Pclass
- Age

---

## Evaluation
- Confusion Matrix used to evaluate model performance
- Random Forest showed better classification results

---

## Key Insights
- Gender was the strongest survival factor
- Passenger class and fare also had strong influence
- Survival was not random; social class played a major role

---

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

##  Future Improvements
- Hyperparameter tuning
- Advanced models (XGBoost)
- Deployment using Streamlit
