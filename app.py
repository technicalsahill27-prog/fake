import streamlit as st
import joblib

# Load model + vectorizer
model, vectorizer = joblib.load("fake_review_model.pkl")

def predict_review(review):
    review_vectorized = vectorizer.transform([review])
    result = model.predict(review_vectorized)
    if result[0] == 1:
        return "✅ Genuine Review"
    else:
        return "⚠️ Fake Review"

st.title("Fake Review Detection App 📝")
st.write("Enter a product review below to check if it's **Fake** or **Genuine**.")

user_input = st.text_area("Write your review here:")

if st.button("Predict"):
    if user_input.strip() != "":
        prediction = predict_review(user_input)
        st.subheader("Result:")
        st.write(prediction)
    else:
        st.warning("⚠️ Please enter a review before predicting.")