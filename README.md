# NEXA Sentiment Analysis System
The NEXA Sentiment Analysis System is a real-time Twitter sentiment analysis application that leverages machine learning techniques to predict the sentiment of tweets. Built using Python and Streamlit, the application offers an interactive web interface where users can input tweet text and immediately view sentiment predictions along with detailed visualizations. The complete pipeline—from data preprocessing and exploratory data analysis (EDA) to model training and evaluation—is provided in the accompanying Jupyter Notebook.

## Features
- **Real-Time Sentiment Analysis:** Enter tweet text to get instant sentiment predictions.
- **Interactive Visualizations:** Includes bar charts displaying prediction probabilities and gauge charts showing prediction confidence.
- **Robust Preprocessing:** Implements text cleaning (e.g., lowercasing, punctuation removal, tokenization, stopword removal) to prepare raw tweet data.
- **Model Comparison:** Trains and compares multiple models (Logistic Regression, Random Forest, Deep LSTM) and selects the best based on evaluation metrics.
- **Session Analytics:** Maintains a prediction history during a session, providing session-level insights.

## Model Details
- **Preprocessing:** Input text is converted to lowercase, numbers and punctuation are removed, extra whitespace is trimmed, and English stopwords are filtered out.
- **Vectorization:** A TF-IDF vectorizer transforms the cleaned text into numerical features.
- **Model Comparison:**  Three models were trained:
   - 1. Logistic Regression: Moderate performance.
   - 2. Random Forest: Achieved the highest F1 Score $(≈ 0.89)$ and is used for prediction.
   - 3. Deep LSTM: Demonstrated lower performance on this dataset.
- **Inference:** The best performing model `(Random Forest)` is loaded in the app, and predictions are made based on TF-IDF features.

Application is hosted on Streamlit Cloud. You can check it out. - [Twitter Sentiment Analyzer](https://twitter-sentiment-analyzer1.streamlit.app/)

## How to run it on Local Host 
- First, clone your repository from GitHub:
```
git clone https://github.com/KaifAhmad1/sentiment-analyzer.git
```
- Navigate through repo
```
cd sentiment-analyzer
```
- Create a Virtual Environment
```
python -m venv sentiment-env
```
- Activate It.
```
source sentiment-env/bin/activate  # On Windows use `sentiment-env\Scripts\activate`
```
-  Install Dependencies
```
pip install -r requirements.txt
```
- Run Streamlit Application
```
streamlit run app.py 
```

## Repo Structure 
```
sentiment-analyzer/
├── app.py                # Main Streamlit application contain UI Logic
├── utils.py              # Utility functions for text processing, predictions, and plotting
├── NEXA_Sentiment_Analysis_System.ipynb        # Google Colab Notebook with the entire pipeline from data loading to metrics comparision
├── requirements.txt      # Project dependencies
├── README.md             # Project overview and instructions
└── models/               # Folder containing trained models and vectorizers
    ├── best_rf_model.joblib
    └── tfidf_vectorizer.joblib
└── data/               # Folder containing training and testing data download from kaggle for twitter sentiment analysis.
    ├── twitter_training.csv
    └── twitter_validation.csv
├── Images             # Containing EDA and Data Analysis Graphs and Charts. 
```

### Author: Mohd Kaif 
