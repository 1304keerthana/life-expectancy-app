import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("🌍 Life Expectancy Predictor")

# Upload dataset
file = st.file_uploader("Upload Life Expectancy CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select features
    features = ['Adult Mortality', 'Alcohol', 'GDP', 'Schooling', 'HIV/AIDS']
    target = 'Life expectancy '

    df = df[features + [target]].dropna()

    X = df[features]
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    st.subheader("📊 Model Performance")
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("R² Score:", r2_score(y_test, y_pred))

    # Coefficients
    st.subheader("📈 Feature Importance")
    coef_df = pd.DataFrame(model.coef_, features, columns=["Coefficient"])
    st.write(coef_df)

    # Plot
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    st.pyplot(plt)
