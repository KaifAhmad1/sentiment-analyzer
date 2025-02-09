import re
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    """
    Clean input text by converting to lowercase, removing digits, punctuation, extra whitespace,
    and filtering out English stopwords. Returns the cleaned text and token list.
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens), tokens

def predict_sentiment(text, model, vectorizer):
    """
    Cleans and vectorizes the input text, then uses the loaded model to predict its sentiment.
    Returns the predicted sentiment, probabilities, cleaned text, and token list.
    """
    cleaned, tokens = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    return prediction, probabilities, cleaned, tokens

def plot_probabilities(probabilities, model):
    """
    Returns a Plotly bar chart showing the prediction probabilities for each sentiment class.
    """
    labels = model.classes_
    df = pd.DataFrame({'Sentiment': labels, 'Probability': probabilities})
    fig = px.bar(
        df,
        x='Sentiment',
        y='Probability',
        title="Prediction Probabilities",
        color='Sentiment',
        range_y=[0, 1]
    )
    return fig

def plot_gauge(confidence, sentiment):
    """
    Returns a Plotly gauge chart that visualizes the prediction confidence.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': f"Confidence for {sentiment}"},
        gauge={'axis': {'range': [0, 1]}}
    ))
    fig.update_layout(margin={'t': 50, 'b': 0, 'l': 0, 'r': 0})
    return fig
