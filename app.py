import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# LOAD MODEL
# ======================
model = pickle.load(open("titanic_model.pkl", "rb"))

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# ======================
# CLEAN MODERN UI STYLE
# ======================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #f5f7fa, #c3cfe2);
    color: #111;
}

/* Title */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #0d47a1;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #333;
}

/* Button */
.stButton button {
    background-color: #0d47a1;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================
st.markdown('<div class="title">🚢 Titanic Survival Prediction AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning Web App (Clean & Modern UI)</div>', unsafe_allow_html=True)
st.markdown("---")

# ======================
# SIDEBAR MENU
# ======================
menu = st.sidebar.selectbox("📌 Menu", ["Prediction", "Analytics", "About"])

# ======================
# PAGE 1 - PREDICTION
# ======================
if menu == "Prediction":

    st.subheader("👤 Passenger Details")

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 0, 100, 25)
        fare = st.number_input("Fare", 0.0, 500.0, 50.0)

    with col2:
        sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
        parch = st.number_input("Parents/Children", 0, 10, 0)
        embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])

    # ======================
    # FEATURE ENGINEERING
    # ======================
    sex = 1 if sex == "Female" else 0

    embarked_Q = 1 if embarked == "Q" else 0
    embarked_S = 1 if embarked == "S" else 0

    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    input_data = np.array([[
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked_Q,
        embarked_S,
        family_size,
        is_alone
    ]])

    # ======================
    # PREDICTION
    # ======================
    if st.button("🚀 Predict Survival"):

        prediction = model.predict(input_data)[0]

        # probability check (only if model supports it)
        try:
            proba = model.predict_proba(input_data)[0][1]
        except:
            proba = None

        st.markdown("### 🧾 Result")

        if proba is not None:
            st.progress(int(proba * 100))
            st.write(f"🔥 Survival Probability: {proba*100:.2f}%")

        if prediction == 1:
            st.success("🎉 Passenger SURVIVED")
            st.balloons()
        else:
            st.error("💀 Passenger DID NOT survive")

# ======================
# PAGE 2 - ANALYTICS
# ======================
elif menu == "Analytics":

    st.subheader("📊 Simple Survival Insights")

    data = pd.DataFrame({
        "Group": ["Female", "Male", "Class 1", "Class 3"],
        "Survival Rate": [0.75, 0.20, 0.62, 0.25]
    })

    fig, ax = plt.subplots()
    sns.barplot(x="Group", y="Survival Rate", data=data, ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)

    st.info("💡 Females and 1st class passengers had higher survival chances.")

# ======================
# PAGE 3 - ABOUT
# ======================
elif menu == "About":

    st.subheader("📘 About Project")

    st.write("""
    🚢 Titanic Survival Prediction using Machine Learning

    ✔ Model: Random Forest  
    ✔ Input: Passenger features  
    ✔ Output: Survival prediction  

    💡 Skills:
    - Data Cleaning
    - Feature Engineering
    - ML Model Training
    - Streamlit Deployment
    """)

    st.success("Built for ML Portfolio 🚀")