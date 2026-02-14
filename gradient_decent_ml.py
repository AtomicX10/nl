import streamlit as st
import numpy as np
import pandas as pd


def sig(x):
    return 1 / (1 + np.exp(-x))

def sig_der(x):
    return x * (1 - x)


def train(X, y, epochs=1000, lr=0.1):
    input_nodes = X.shape[1]
    hidden_nodes = 4
    output_nodes = 1
    
    # Init weights (w1, w2) and biases (b1, b2)
    np.random.seed(42)
    w1 = np.random.rand(input_nodes, hidden_nodes)
    b1 = np.random.rand(1, hidden_nodes)
    w2 = np.random.rand(hidden_nodes, output_nodes)
    b2 = np.random.rand(1, output_nodes)
    
    logs = []
    losses = []


    for i in range(epochs):

        z1 = np.dot(X, w1) + b1
        l1 = sig(z1)
        
        z2 = np.dot(l1, w2) + b2
        l2 = sig(z2)
    
        error = y - l2

        d_l2 = error * sig_der(l2)
        
        error_l1 = d_l2.dot(w2.T)
        d_l1 = error_l1 * sig_der(l1)
        
        w2 += l1.T.dot(d_l2) * lr
        b2 += np.sum(d_l2, axis=0, keepdims=True) * lr
        w1 += X.T.dot(d_l1) * lr
        b1 += np.sum(d_l1, axis=0, keepdims=True) * lr
        
        if i % 100 == 0:
            current_loss = np.mean(np.abs(error))
            losses.append(current_loss)

            logs.append(f"Epoch {i}: Loss {current_loss:.4f}")

    return w1, b1, w2, b2, logs, losses

def predict(vals, w1, b1, w2, b2):
    z1 = np.dot(vals, w1) + b1
    l1 = sig(z1)
    z2 = np.dot(l1, w2) + b2
    l2 = sig(z2)
    return l2

st.title("Simple NN Implementation")

st.subheader("Dataset")
upl_file = st.file_uploader("Upload CSV", type="csv")

if upl_file:
    df = pd.read_csv(upl_file)
else:
    st.info("Using default XOR data")
    df = pd.DataFrame({
        'A': [0, 0, 1, 1],
        'B': [0, 1, 0, 1],
        'Y': [0, 1, 1, 0]
    })

st.dataframe(df)

# Prep data
X_train = df.iloc[:, :-1].values
y_train = df.iloc[:, -1].values.reshape(-1, 1)

# 2. Training
st.subheader("Training Params")
col1, col2 = st.columns(2)
ep = col1.number_input("Epochs", value=1000, step=100)
rate = col2.number_input("Learning Rate", value=0.1)

if st.button("Train Model"):
    w1, b1, w2, b2, history, losses = train(X_train, y_train, ep, rate)

    
    # Store weights in session
    st.session_state['w1'] = w1
    st.session_state['b1'] = b1
    st.session_state['w2'] = w2
    st.session_state['b2'] = b2
    st.session_state['trained'] = True
    
    st.success("Done!")
    st.subheader("Epoch vs Loss")
    st.line_chart(losses)
    st.text("Updates every 100 epochs:")
    st.text("\n".join(history))
    
    st.write("Final Weights (Hidden->Output):")
    st.write(w2)

st.subheader("Test Values")

if 'trained' in st.session_state:
    user_inputs = []
    cols = st.columns(len(df.columns) - 1)
    
    for idx, col in enumerate(df.columns[:-1]):
        val = cols[idx].number_input(f"Val for {col}", value=0.0)
        user_inputs.append(val)
        
    if st.button("Predict"):
        test_arr = np.array([user_inputs])
        res = predict(test_arr, st.session_state['w1'], st.session_state['b1'], st.session_state['w2'], st.session_state['b2'])
        
        st.write(f"Raw Output: {res[0][0]:.4f}")
        st.metric("Prediction", "1" if res > 0.5 else "0")
else:
    st.warning("Train the model first to see predictions.")

