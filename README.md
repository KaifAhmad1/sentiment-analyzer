# sentiment-analyzer
### How to run it on Local Host 
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

### Repo Structure 
```
sentiment-analyzer/
├── app.py                # Main Streamlit application
├── utils.py              # Utility functions for text processing, predictions, and plotting
├── NEXA_Sentiment_Analysis_System.ipynb        # Jupyter Notebook with the entire pipeline
├── requirements.txt      # Project dependencies
├── README.md             # Project overview and instructions
└── models/               # Folder containing trained models and vectorizers
    ├── best_rf_model.joblib
    └── tfidf_vectorizer.joblib
```
