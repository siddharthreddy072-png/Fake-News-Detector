import streamlit as st
import joblib

# Load the trained model
model = joblib.load("model/fake_news_model.pkl")

# Page title
st.title("📰 Fake News Detector")

st.write("Enter a news article below to check its predicted category.")

# Text input
news_text = st.text_area(
    "Enter News Article:",
    height=200
)

# Prediction button
if st.button("Check News"):

    if news_text.strip() == "":
        st.warning("Please enter a news article.")

    else:
        # Predict the news category
        prediction = model.predict([news_text])[0]

        if str(prediction).upper() == "FAKE":
            st.error("❌ This news is predicted to be FAKE.")

        else:
            st.success("✅ This news is predicted to be REAL.")
