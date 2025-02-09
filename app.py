import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from utils import clean_text, predict_sentiment, plot_probabilities, plot_gauge
import nltk

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load the trained model and TF-IDF vectorizer
try:
    model = joblib.load('models/best_rf_model.joblib')
    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

    # Check if the vectorizer is fitted
    if not hasattr(vectorizer, 'vocabulary_'):
        st.error("Vectorizer is not fitted. Please check the training process.")
    else:
        st.success("Vectorizer is fitted and ready to use.")
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")

# Initialize session state to store prediction history
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar: Session Insights and History
st.sidebar.title("Session Insights")
if st.sidebar.button("Clear History"):
    st.session_state.history = []
    st.sidebar.success("Session history cleared!")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.sidebar.subheader("Sentiment Distribution")
    pie_fig = px.pie(history_df, names='Predicted Sentiment', title="Session Sentiment Distribution")
    st.sidebar.plotly_chart(pie_fig)
    st.sidebar.subheader("Prediction History")
    st.sidebar.dataframe(history_df)

# Main App UI
st.title("Enhanced Real-Time Twitter Sentiment Analysis")
st.write("Enter a tweet below to analyze its sentiment and gain real-time insights.")

# User input for tweet text
user_input = st.text_area("Enter Tweet Text", placeholder="Type your tweet here...", height=100)

if st.button("Analyze Sentiment"):
    if user_input:
        try:
            prediction, probabilities, cleaned_text, tokens = predict_sentiment(user_input, model, vectorizer)
            confidence = np.max(probabilities)

            st.subheader("Prediction")
            st.write(f"**Predicted Sentiment:** {prediction}")
            st.write(f"**Prediction Confidence:** {confidence:.2f}")

            st.subheader("Cleaned Input Text")
            st.write(cleaned_text)
            st.write(f"**Token Count:** {len(tokens)}")

            st.subheader("Prediction Probabilities")
            fig_prob = plot_probabilities(probabilities, model)
            st.plotly_chart(fig_prob)

            st.subheader("Confidence Gauge")
            fig_gauge = plot_gauge(confidence, prediction)
            st.plotly_chart(fig_gauge)

            # Save current prediction details to session history
            st.session_state.history.append({
                "Original Text": user_input,
                "Cleaned Text": cleaned_text,
                "Token Count": len(tokens),
                "Predicted Sentiment": prediction,
                "Confidence": round(confidence, 2)
            })
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.markdown("### How It Works")
st.write("""
1. **Processing:** Your tweet is cleaned and tokenized.
2. **Vectorization:** The cleaned text is transformed using a pre-trained TF-IDF vectorizer.
3. **Prediction:** A Random Forest model predicts the sentiment and outputs a probability distribution.
4. **Visualization:**
   - **Bar Chart:** Displays prediction probabilities.
   - **Gauge Chart:** Shows the prediction confidence.
5. **Session Analytics:** Aggregates predictions and displays a session-based sentiment distribution.
""")
