import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Title
st.title("📩 SMS Spam Detection using Lasso Regression")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

df = load_data()

# Preview dataset
st.subheader("🔍 Dataset Preview")
st.write(df.head())

# Statistical Analysis
st.subheader("📊 Statistical Summary")
st.write(df.describe())

# Missing Values
st.subheader("❗ Missing Values")
st.write(df.isnull().sum())

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])  # ham=0, spam=1

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

st.subheader("🧮 TF-IDF Feature Count")
st.write("Total Features Created:", X.shape[1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train Lasso
def train_lasso(alpha):
    model = Lasso(alpha=alpha)
    model.fit(X_train.toarray(), y_train)
    return model

# Train models with different alpha
alphas = [0.01, 0.1, 1]
results = {}

for alpha in alphas:
    model = train_lasso(alpha)
    coef = model.coef_

    non_zero = np.sum(coef != 0)
    zero = np.sum(coef == 0)

    results[alpha] = (non_zero, zero)

# Display results
st.subheader("📉 Feature Selection using Lasso")

for alpha in alphas:
    st.write(f"Alpha = {alpha}")
    st.write(f"Non-zero features: {results[alpha][0]}")
    st.write(f"Eliminated features: {results[alpha][1]}")
    st.write("---")

# Percentage reduction (alpha=0.1)
original_features = X.shape[1]
selected_features = results[0.1][0]

reduction = ((original_features - selected_features) / original_features) * 100

st.subheader("📉 Feature Reduction")
st.write(f"Percentage Reduction: {reduction:.2f}%")

# Prediction on test data
model = train_lasso(0.1)
y_pred = model.predict(X_test.toarray())
y_pred = np.where(y_pred > 0.5, 1, 0)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("🎯 Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

# User Input Prediction
st.subheader("✉️ Predict Your Own Message")

user_input = st.text_area("Enter SMS message:")

if st.button("Predict"):
    if user_input:
        user_vec = vectorizer.transform([user_input])
        pred = model.predict(user_vec.toarray())
        pred_label = "Spam" if pred[0] > 0.5 else "Ham"
        st.success(f"Prediction: {pred_label}")
    else:
        st.warning("Please enter a message!")
