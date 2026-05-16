import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page settings
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")

st.write("Enter news text and detect whether it is FAKE or REAL.")

# User input
news_input = st.text_area("Enter News Text")

if st.button("Detect News"):

    if news_input.strip() != "":

        # Transform text
        transformed_text = vectorizer.transform([news_input])

        # Prediction
        prediction = model.predict(transformed_text)[0]

        # Probability
        probability = model.predict_proba(transformed_text)

        confidence = round(max(probability[0]) * 100, 2)

        # Result
        if prediction == "FAKE":
            st.error(f"❌ Fake News Detected")

        else:
            st.success(f"✅ Real News Detected")

        st.info(f"Confidence Score: {confidence}%")

    else:
        st.warning("Please enter news text.")