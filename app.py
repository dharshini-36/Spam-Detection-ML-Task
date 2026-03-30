import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Title
# -------------------------------
st.title("📩 SMS Spam Detection using Lasso + Logistic Regression")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

df = load_data()

# -------------------------------
# Dataset Preview
# -------------------------------
st.subheader("🔍 Dataset Preview")
st.write(df.head())

# -------------------------------
# Statistical Summary
# -------------------------------
st.subheader("📊 Statistical Summary")
st.write(df.describe())

# -------------------------------
# Missing Values
# -------------------------------
st.subheader("❗ Missing Values")
st.write(df.isnull().sum())

# -------------------------------
# Encode Labels
# -------------------------------
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])  # ham=0, spam=1

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['message'])
y = df['label']

st.subheader("🧮 TF-IDF Features")
st.write("Total Features Created:", X.shape[1])

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Lasso Feature Selection Function
# -------------------------------
def lasso_feature_selection(alpha):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train.toarray(), y_train)

    coef = lasso.coef_
    selected = coef != 0

    non_zero = np.sum(selected)
    zero = np.sum(coef == 0)

    return selected, non_zero, zero

# -------------------------------
# Compare Alpha Values
# -------------------------------
st.subheader("📉 Feature Selection using Lasso")

alphas = [0.0001, 0.001, 0.01]
results = {}

for alpha in alphas:
    selected, non_zero, zero = lasso_feature_selection(alpha)
    results[alpha] = (selected, non_zero, zero)

    st.write(f"Alpha = {alpha}")
    st.write(f"Selected Features: {non_zero}")
    st.write(f"Eliminated Features: {zero}")
    st.write("---")

# -------------------------------
# Feature Reduction %
# -------------------------------
original_features = X.shape[1]
selected_features = results[0.0001][1]

reduction = ((original_features - selected_features) / original_features) * 100

st.subheader("📉 Feature Reduction (alpha = 0.0001)")
st.write(f"Percentage Reduction: {reduction:.2f}%")

# -------------------------------
# Select Best Features
# -------------------------------
selected_features_mask = results[0.0001][0]

# ✅ SAFETY CHECK
if np.sum(selected_features_mask) == 0:
    st.error("⚠️ All features removed by Lasso! Using original features instead.")
    X_train_selected = X_train
    X_test_selected = X_test
    selected_feature_names = vectorizer.get_feature_names_out()
else:
    X_train_selected = X_train[:, selected_features_mask]
    X_test_selected = X_test[:, selected_features_mask]
    selected_feature_names = vectorizer.get_feature_names_out()[selected_features_mask]

# -------------------------------
# Logistic Regression Model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_selected, y_train)

# -------------------------------
# Prediction on Test Data
# -------------------------------
y_pred = model.predict(X_test_selected)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("🎯 Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

# -------------------------------
# Confusion Matrix
# -------------------------------
st.subheader("📊 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

# -------------------------------
# Top Important Words
# -------------------------------
st.subheader("🔥 Top Important Words")

coefficients = model.coef_[0]

top_indices = np.argsort(np.abs(coefficients))[-10:]

top_words = [(selected_feature_names[i], coefficients[i]) for i in top_indices]

st.write(top_words)

# -------------------------------
# User Input Prediction
# -------------------------------
st.subheader("✉️ Predict Your Own Message")

user_input = st.text_area("Enter SMS message:")

if st.button("Predict"):
    if user_input:
        user_vec = vectorizer.transform([user_input])

        if np.sum(selected_features_mask) == 0:
            user_vec_selected = user_vec
        else:
            user_vec_selected = user_vec[:, selected_features_mask]

        pred = model.predict(user_vec_selected)[0]
        label = "Spam" if pred == 1 else "Ham (Not Spam)"

        st.success(f"Prediction: {label}")
    else:
        st.warning("Please enter a message!")
