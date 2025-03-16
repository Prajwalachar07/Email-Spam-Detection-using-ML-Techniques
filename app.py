# Install required libraries
# pip install streamlit pandas scikit-learn nltk

from data_generator import generate_synthetic_data
from flask import Flask,render_template,request,redirect,url_for,flash
import pandas as pd
import numpy as np
import string
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import mysql.connector

app =Flask(__name__)
app.secret_key='bdc8cb090e51a3ab1f4c6e2d2e35735b'
nltk.download('stopwords')
nltk.download('wordnet')

SPAM_EMOJIS = ["🎉", "🔥", "💰", "🤑", "📢", "‼️","🤯","🚀","🔎","📈","👋"]
NON_SPAM_EMOJIS = ["😊", "🎂", "🙏", "👍", "📧"]



def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()  # Convert to lowercase
    text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation

    # Adjust emoji extraction based on the library version
    try:
        emoji_list = [char['emoji'] for char in emoji.emoji_list(text)]
    except AttributeError:
        # Fallback for older versions: Check each character
        emoji_list = [char for char in text if char in emoji.EMOJI_DATA]

    # Remove emojis from text
    text = "".join([char for char in text if char not in emoji_list])

    # Lemmatize and remove stopwords
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]

    return " ".join(words), emoji_list





# Function to vectorize messages
def vectorize_data(data):
    vectorizer = TfidfVectorizer()
    X_text = vectorizer.fit_transform(data['processed_message'])

    # Add emoji counts as features
    emoji_features = np.array(data['emoji_counts'].tolist())
    emoji_features = np.array(emoji_features)
    combined_features = np.hstack((X_text.toarray(), emoji_features))
    
    return combined_features, vectorizer

# Train the model
def train_model(data):
    X, vectorizer = vectorize_data(data)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    classification = classification_report(y_test, y_pred)

    with open('spam_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    return acc, classification
@app.route("/")  # Root route
def index():
    return render_template("index.html")
@app.route("/home")
def home():
    return render_template("home.html")



@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        # Load and preprocess data
        data = generate_synthetic_data()
        data['processed_message'], data['extracted_emojis'] = zip(*data['message'].apply(preprocess_text))
        data['emoji_counts'] = data['extracted_emojis'].apply(lambda emojis: [emojis.count(e) for e in SPAM_EMOJIS + NON_SPAM_EMOJIS])

        # Train the model
        acc, classification = train_model(data)

        # Render the training page with the results
        return render_template('train.html', accuracy=acc, report=classification)

    return render_template('train.html', accuracy=None, report=None)


@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        message = request.form['message']
        with open('spam_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    
        processed_message, emojis = preprocess_text(message)
        emoji_counts = [emojis.count(emoji) for emoji in SPAM_EMOJIS + NON_SPAM_EMOJIS]
        transformed_message = vectorizer.transform([processed_message]).toarray()
        combined_features = np.hstack((transformed_message, np.array([emoji_counts])))
    
        prediction = model.predict(combined_features)
        result ="Spam" if prediction[0] == 1 else "Not Spam"
        return render_template('test.html', message=message, result=result)
    return render_template('test.html',result=None)

if __name__ == "__main__":
    print("Available Routes:")
    print(app.url_map)
    app.run(debug=False)
