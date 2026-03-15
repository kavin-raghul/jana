# YouTube Comment Topic Classification

A real-time NLP project that classifies YouTube comments into four categories:
- Funny
- Spam
- Informative
- Hate

## Project Structure

```
project/
├── data/
│   └── dataset.csv         # Sample dataset
├── src/
│   ├── preprocess.py       # Text cleaning and preprocessing (NLTK)
│   ├── train_model.py      # Model training & evaluation
│   └── predictor.py        # Inference class for predictions
├── model/
│   ├── classifier.pkl      # Saved trained model
│   └── vectorizer.pkl      # Saved TF-IDF vectorizer
├── app.py                  # Streamlit web application
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Setup Instructions

1. **Clone or navigate to the project repository.**
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Train the machine learning model:**
   This step processes the dataset, trains multiple models (Naive Bayes, Logistic Regression, SVM), selects the best one, and saves it to the `model/` directory.
   ```bash
   python src/train_model.py
   ```

## Running the Web App

Once the model is trained and saved, you can launch the real-time prediction interface using Streamlit:

```bash
streamlit run app.py
```

This will open a web browser where you can enter a comment and see the predicted category instantly.

## How it Works

1. **Preprocessing**: The raw text undergoes lowercasing, punctuation removal, tokenization, stopword removal, and lemmatization using NLTK.
2. **Feature Engineering**: The cleaned text is converted into numerical features using a TF-IDF Vectorizer.
3. **Classification**: A trained machine learning model predicts the category based on the numerical features.
