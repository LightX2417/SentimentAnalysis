import streamlit as st
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Load the saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing functions
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()


def preprocess(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    tokens = word_tokenize(text)
    tokens = [
        ps.stem(word.lower())
        for word in tokens
        if word.isalpha() and word.lower() not in stop_words
    ]
    return " ".join(tokens)



def predict_sentiment(text):
    processed_text = preprocess(text)
    text_vectorized = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(text_vectorized)
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return sentiment


# Streamlit app
st.title("Twitter Sentiment Analysis")
st.write("Enter text to analyze the sentiment.")


tweet_text = st.text_area("Enter the tweet text:")
if tweet_text:
    sentiment = predict_sentiment(tweet_text)
    st.write(f"Sentiment: {sentiment}")
