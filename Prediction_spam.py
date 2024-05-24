import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import load_model

# Load the trained model
model = load_model('spam_detection_model.h5')

# Load the fitted vectorizer
import joblib
feature_extraction = joblib.load('tfidf_vectorizer.pkl')

# Function to preprocess the input text
def preprocess_text(text):
    # Text vectorization
    X_features = feature_extraction.transform([text]).toarray()
    return X_features

# Function to classify email as spam or ham
def classify_email(text):
    # Preprocess the input text
    X_input = preprocess_text(text)
    # Predict using the trained model
    prediction = model.predict(X_input)
    # Classify as spam (0) or ham (1)
    if prediction[0] > 0.5:
        return "Ham"
    else:
        return "Spam"

# Example usage
input_email = input("Enter the email message: ")
result = classify_email(input_email)
print("Predicted result:", result)