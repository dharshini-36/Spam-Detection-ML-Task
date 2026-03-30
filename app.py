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
# TF-IDF Vectorization (Improved)
# -------------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2)   # 🔥 important improvement
)

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
# Lasso Feature Selection (for report)
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
# FINAL MODEL (No Lasso for prediction)
# -------------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'   # 🔥 handles imbalance
)

model.fit(X_train, y_train)

# -------------------------------
# Prediction on Test Data
# -------------------------------
y_pred = model.predict(X_test)

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

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

top_indices = np.argsort(np.abs(coefficients))[-10:]

top_words = [(feature_names[i], coefficients[i]) for i in top_indices]

# -------------------------------
# User Input Prediction
# -------------------------------
st.subheader("✉️ Predict Your Own Message")

user_input = st.text_area("Enter SMS message:")

if st.button("Predict"):
    if user_input:
        user_vec = vectorizer.transform([user_input])

        # Probability-based prediction
        prob = model.predict_proba(user_vec)[0][1]

        if prob > 0.4:
            label = "Spam"
        else:
            label = "Ham"

        # 🎨 Colored Output
        if label == "Spam":
            st.markdown(
                f"<div style='background-color:#ff4d4d;padding:15px;border-radius:10px;color:white;font-size:18px;'>"
                f"🚨 Prediction: SPAM<br>Probability: {prob:.2f}"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='background-color:#4CAF50;padding:15px;border-radius:10px;color:white;font-size:18px;'>"
                f"✅ Prediction: HAM (Not Spam)<br>Probability: {prob:.2f}"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.warning("Please enter a message!")
