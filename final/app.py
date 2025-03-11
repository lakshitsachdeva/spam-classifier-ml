import subprocess
import sys
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords', force=True)

import pickle
import string
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# 🔹 Ensure required NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError: 
    nltk.download('stopwords')

# 🔹 Initialize Porter Stemmer
ps = PorterStemmer()

# 🔹 Text Preprocessing Function
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenization
    y = [i for i in text if i.isalnum()]  # Remove special characters

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Apply stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# 🔹 Load Model and Vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model or vectorizer file not found! Please check file paths.")

# 🔹 Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message:")

if st.button('Predict'):
    if input_sms.strip():
        # Preprocess input
        transformed_sms = transform_text(input_sms)

        # Vectorize input
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Display result
        st.header("Spam" if result == 1 else "Not Spam")
    else:
        st.warning("Please enter a message before predicting.")

st.header("Made by Lakshit Sachdeva")
