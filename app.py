import streamlit as st
import os

# Set page config
st.set_page_config(
    page_title="YouTube Comment Classifier",
    page_icon="📺",
    layout="centered"
)

# Function to load predictor
@st.cache_resource
def load_predictor():
    try:
        from src.predictor import CommentPredictor
        return CommentPredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Have you trained the model yet? Run `python src/train_model.py` first.")
        return None

# Load the predictor
predictor = load_predictor()

# App UI
st.title("📺 YouTube Comment Topic Classifier")
st.write("""
Enter a YouTube comment below to classify it into one of the following categories:
**Funny**, **Spam**, **Informative**, or **Hate**.
""")

# Input text area
user_input = st.text_area("Enter YouTube comment:", height=100, placeholder="e.g., This tutorial was really helpful, thanks!")

# Predict button
if st.button("Predict Category", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a comment to classify.")
    elif predictor is None:
        st.error("Model is not available. Please check the setup.")
    else:
        with st.spinner("Analyzing comment..."):
            try:
                # Get prediction
                prediction = predictor.predict(user_input)
                
                # Display result with appropriate color/emoji
                if prediction == "Funny":
                    st.success(f"### Predicted Category: 😂 {prediction}")
                elif prediction == "Spam":
                    st.warning(f"### Predicted Category: 🚫 {prediction}")
                elif prediction == "Informative":
                    st.info(f"### Predicted Category: 🧠 {prediction}")
                elif prediction == "Hate":
                    st.error(f"### Predicted Category: 🤬 {prediction}")
                else:
                    st.write(f"### Predicted Category: {prediction}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.write("""
    This is an NLP project that uses Machine Learning to classify YouTube comments.
    
    **Pipeline:**
    1. Text Preprocessing (NLTK)
    2. TF-IDF Vectorization
    3. Classification Model (e.g., Logistic Regression/SVM)
    """)
