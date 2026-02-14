import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ------------------ Utility ------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ------------------ Load Dataset ------------------
data = pd.read_csv("student_data.csv")

X = data[["StudyHours", "SleepHours"]].values
y = data["Pass"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ------------------ Streamlit UI ------------------
st.title("🎓 Student Pass Prediction (Neural Network From Scratch)")

st.sidebar.header("Hyperparameters")
lr = st.sidebar.slider("Learning Rate", 0.001, 0.5, 0.1)
epochs = st.sidebar.slider("Epochs", 100, 2000, 500)

# ------------------ Initialize Weights ------------------
np.random.seed(1)

# 2 inputs → 2 hidden → 1 output
w1, w2, w3, w4 = np.random.randn(4) * 0.1
w5, w6 = np.random.randn(2) * 0.1

bh1, bh2, bo = 0.0, 0.0, 0.0

# ------------------ Training ------------------
for epoch in range(epochs):
    for i in range(len(X)):
        x1, x2 = X[i]

        # Forward pass
        zh1 = x1*w1 + x2*w3 + bh1
        zh2 = x1*w2 + x2*w4 + bh2

        h1 = sigmoid(zh1)
        h2 = sigmoid(zh2)

        zo = h1*w5 + h2*w6 + bo
        o = sigmoid(zo)

        # Backprop
        error = y[i] - o

        delta_o = error * o * (1 - o)
        delta_h1 = delta_o * w5 * h1 * (1 - h1)
        delta_h2 = delta_o * w6 * h2 * (1 - h2)

        # Update weights
        w5 += lr * delta_o * h1
        w6 += lr * delta_o * h2
        bo += lr * delta_o

        w1 += lr * delta_h1 * x1
        w3 += lr * delta_h1 * x2
        w2 += lr * delta_h2 * x1
        w4 += lr * delta_h2 * x2

        bh1 += lr * delta_h1
        bh2 += lr * delta_h2

# ------------------ Prediction UI ------------------
st.subheader("📘 Enter Student Details")

study = st.number_input("Study Hours", value=5.0)
sleep = st.number_input("Sleep Hours", value=7.0)

input_data = scaler.transform([[study, sleep]])
x1, x2 = input_data[0]

# Forward pass
zh1 = x1*w1 + x2*w3 + bh1
zh2 = x1*w2 + x2*w4 + bh2

h1 = sigmoid(zh1)
h2 = sigmoid(zh2)

zo = h1*w5 + h2*w6 + bo
o = sigmoid(zo)

# ------------------ Output ------------------
st.subheader("📊 Prediction")

st.write("Pass Probability:", round(float(o), 4))
st.write("Result:", "✅ Pass" if o >= 0.5 else "❌ Fail")
