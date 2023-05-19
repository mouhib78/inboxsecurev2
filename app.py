from flask import Flask, render_template, request,jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import Prediction as pr
import re

app = Flask(__name__)


def extract_links(text):
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    matches = re.findall(pattern, text)
    return matches

def spam_prediction(text):
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

    # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']

    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    # Predict the text
    data = [text]
    vect = cv.transform(data).toarray()
    prediction = clf.predict(vect)

    return prediction[0]


@app.route('/')
def home():
     return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
     message = request.form['message']
     spam_result = spam_prediction(message)
     sentiment_result = pr.prediction(message)
     return render_template('index.html', spam_prediction=spam_result, sentiment_prediction=sentiment_result)




if __name__ == '__main__':
        app.run()
