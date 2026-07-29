import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
import pickle
import time
import numpy as np

# -------------------- PAGE TITLE --------------------
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction Using Machine Learning")

st.image("https://miro.medium.com/v2/resize:fit:1400/0*cDRFtpTiOJFrfzS5.jpg")

st.write("""
### Welcome to Zytexa's House Price Prediction System

This AI-powered application predicts house prices quickly and accurately based on important property features.
""")

# -------------------- ABOUT PROJECT --------------------
with st.expander("📖 About Project"):
    st.write("""
This application uses a Machine Learning model trained on the California Housing Dataset
to estimate house prices based on user inputs.
""")

# -------------------- DATASET --------------------
data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["MedHouseVal"] = data.target

X = df.iloc[:, :-2]

scaler = MinMaxScaler()
scaler.fit(X)

# -------------------- SIDEBAR --------------------
st.sidebar.title("🏠 House Price Prediction")
st.sidebar.markdown("### Developed by Rahul Singh")

st.sidebar.image(
    "https://i.pinimg.com/originals/b1/cf/53/b1cf530756b23930aab499c1a11b443c.gif"
)

st.sidebar.markdown("### Enter House Details")

all_value = []

for i in X.columns:
    value = st.sidebar.slider(
        f"{i}",
        float(X[i].min()),
        float(X[i].max()),
        float(X[i].mean())
    )
    all_value.append(value)

final_data = scaler.transform([all_value])

# -------------------- LOAD MODEL --------------------
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict House Price"):

    with st.spinner("Predicting House Price..."):
        time.sleep(2)

        prediction = model.predict(final_data)[0]

    st.success(f"🏡 Estimated House Price: ${abs(prediction*100000):,.2f}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("© 2026 Rahul Singh | House Price Prediction using Machine Learning")