import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and preprocesses the input text.
    Steps:
    1. Lowercase
    2. Remove punctuation and special characters
    3. Tokenization
    4. Remove stopwords
    5. Lemmatization
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation and special characters (keep only alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # 3. Tokenization
    # Fallback to simple split if tokenization fails due to un-downloaded punkt
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        tokens = nltk.word_tokenize(text)
    
    # 4 & 5. Remove stopwords and Lemmatize
    clean_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words
    ]
    
    # Join back into a single string
    return " ".join(clean_tokens)

if __name__ == "__main__":
    # Test the preprocessing function
    sample = "This is a SAMPLE comment! It has punctuation, STOPWORDS, and needs lemmatization (running)."
    print("Original:", sample)
    print("Preprocessed:", preprocess_text(sample))
