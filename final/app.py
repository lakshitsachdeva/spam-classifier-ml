import subprocess
import sys

# Ensure nltk is installed
try:
    import nltk
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import shutil
shutil.rmtree('/home/adminuser/nltk_data', ignore_errors=True)
nltk.download('punkt', download_dir='/home/adminuser/nltk_data')

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')

ps= PorterStemmer()


def transform_text(text):
    text = text.lower()  # lower case
    text = nltk.word_tokenize(text)  # tokenization
    y = []
    for i in text:
        if i.isalnum():  # removes special characters
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)



tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS spam Classifier")

input_sms= st.text_area("Enter the message: ")

if st.button('predict'):
    #preprocess
    transformed_sms=transform_text(input_sms)
    #vectorize
    vector_input= tfidf.transform([transformed_sms])
    #predict
    result= model.predict(vector_input)[0]

     #display
    if result == 1:
        st.header("spam")
    else:
        st.header("Not Spam")

st.header("Made by Lakshit Sachdeva")