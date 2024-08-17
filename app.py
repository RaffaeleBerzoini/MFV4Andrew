import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Function to perform polynomial regression
def polynomial_regression(degree, X, y):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    return model, mse

# Streamlit app
st.title("Regression Analysis App")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file, delimiter=';')

    # Convert columns to numeric
    df['MFV'] = df['MFV'].str.replace(',', '.').astype(float)
    df['COSTO 2023'] = df['COSTO 2023'].astype(float)
    
    # Drop rows with missing values
    df_clean = df.dropna(subset=['MFV', 'COSTO 2023'])

    # Reshape data
    X = df_clean['MFV'].values.reshape(-1, 1)
    y = df_clean['COSTO 2023'].values
    
    # Perform regressions
    model_linear = LinearRegression()
    model_linear.fit(X, y)
    y_pred_linear = model_linear.predict(X)

    model_2nd, mse_2nd = polynomial_regression(2, X, y)
    model_3rd, mse_3rd = polynomial_regression(3, X, y)

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.scatter(X, y, color='black', label='Data')
    plt.plot(X, y_pred_linear, color='blue', label=f'Linear (MSE: {mean_squared_error(y, y_pred_linear):.2f})')
    
    X_poly_2nd = PolynomialFeatures(degree=2).fit_transform(X)
    y_pred_2nd = model_2nd.predict(X_poly_2nd)
    plt.plot(X, y_pred_2nd, color='green', label=f'2nd Degree (MSE: {mse_2nd:.2f})')
    
    X_poly_3rd = PolynomialFeatures(degree=3).fit_transform(X)
    y_pred_3rd = model_3rd.predict(X_poly_3rd)
    plt.plot(X, y_pred_3rd, color='red', label=f'3rd Degree (MSE: {mse_3rd:.2f})')

    plt.xlabel('MFV')
    plt.ylabel('COSTO 2023')
    plt.title('Regression Analysis: COSTO 2023 vs MFV')
    plt.legend()
    
    # Display the plot
    st.pyplot(plt)

    # Input for prediction
    st.header("Estimate COSTO 2023 based on MFV")
    mfv_input = st.number_input("Enter MFV value:", min_value=float(X.min()), max_value=float(X.max()), step=0.01)

    if mfv_input:
        # Predict using the linear model
        linear_pred = model_linear.predict([[mfv_input]])

        # Predict using the 2nd degree polynomial model
        mfv_poly_2nd = PolynomialFeatures(degree=2).fit_transform([[mfv_input]])
        poly_2nd_pred = model_2nd.predict(mfv_poly_2nd)

        # Predict using the 3rd degree polynomial model
        mfv_poly_3rd = PolynomialFeatures(degree=3).fit_transform([[mfv_input]])
        poly_3rd_pred = model_3rd.predict(mfv_poly_3rd)

        # Display the predictions
        st.write(f"Linear Model Prediction: {linear_pred[0]:.2f}")
        st.write(f"2nd Degree Polynomial Prediction: {poly_2nd_pred[0]:.2f}")
        st.write(f"3rd Degree Polynomial Prediction: {poly_3rd_pred[0]:.2f}")
