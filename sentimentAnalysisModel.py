import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


# Load the Dataset
file_path = 'sentiment140.csv'  
columns = ['target', 'id', 'date', 'flag', 'user', 'text']
data = pd.read_csv(file_path, encoding='ISO-8859-1', names=columns)

# Keep only the relevant columns
data = data[['target', 'text']]

# Convert the target labels (0 for negative, 4 for positive) to binary (0 for negative, 1 for positive)
data['target'] = data['target'].replace(4, 1)

# Reduce dataset size to 10,000 tweets
data = data.sample(n=100000, random_state=42)

# Preprocess the Data
nltk.download('punkt')
nltk.download("punkt_tab")
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def preprocess(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    # Tokenization and stemming
    tokens = word_tokenize(text)
    tokens = [ps.stem(word.lower()) for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return " ".join(tokens)

# Apply preprocessing to each tweet
data['processed_text'] = data['text'].apply(preprocess)

# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features
X = vectorizer.fit_transform(data['processed_text']).toarray()
y = data['target']

# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model using Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Make Predictions on the Test Set
y_pred = model.predict(X_test)

# Evaluate the Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Test the Model with a Sample Tweet
def predict_sentiment(text):
    processed_text = preprocess(text)
    text_vectorized = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(text_vectorized)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    return sentiment

# Save the model
joblib.dump(model, "sentiment_model.pkl")

# Save the vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
