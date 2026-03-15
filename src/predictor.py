import os
import joblib

# Import preprocess_text correctly depending on how this is run
try:
    from src.preprocess import preprocess_text
except ImportError:
    from preprocess import preprocess_text

class CommentPredictor:
    def __init__(self):
        """Loads the trained model and vectorizer on initialization."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, 'model', 'classifier.pkl')
        vectorizer_path = os.path.join(base_dir, 'model', 'vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Model files not found. Please run train_model.py first.")
            
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
    def predict(self, text):
        """
        Predicts the category of a given comment.
        Returns the predicted category (Funny / Spam / Informative / Hate).
        """
        # 1. Preprocess the input text using the same logic as training
        clean_text = preprocess_text(text)
        
        # 2. Handle empty text after preprocessing
        if not clean_text.strip():
            return "Unknown (Text too short after cleaning)"
            
        # 3. Vectorize the text
        text_features = self.vectorizer.transform([clean_text])
        
        # 4. Predict the category
        prediction = self.model.predict(text_features)
        
        return prediction[0]

if __name__ == "__main__":
    # Test the predictor
    try:
        predictor = CommentPredictor()
        test_comment = "Click my link for free money!"
        print(f"Comment: '{test_comment}'")
        print(f"Prediction: {predictor.predict(test_comment)}")
    except FileNotFoundError as e:
        print(e)
