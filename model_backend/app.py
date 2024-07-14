import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import time

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Load the updated model
model_output_path = "nmap_command_classifier.pkl"
model = joblib.load(model_output_path)

# Load the updated dataset
df_output_path = "nmap_commands_updated.json"
with open(df_output_path, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
df = pd.DataFrame(data)

# Preprocessing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if isinstance(text, dict):
        text = json.dumps(text)  # Convert dict to JSON string if necessary
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return ' '.join(cleaned_tokens)

df['Processed_Description'] = df['Description'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Processed_Description'])

# Function to predict and get feedback
def predict_nmap_command(query):
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])
    predicted_command = model.predict(query_vector)
    return predicted_command[0]

def log_feedback(query, predicted_output, feedback):
    with open("feedback_log.txt", "a") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Predicted Output: {predicted_output}\n")
        f.write(f"Feedback: {feedback}\n")
        f.write("="*30 + "\n")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    query = data.get('query')
    print(f"Received query: {query}")
    predicted_output = predict_nmap_command(query)
    return jsonify({'predicted_command': 'nmap '+predicted_output})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    query = data.get('query')
    predicted_output = data.get('predicted_output')
    feedback = data.get('feedback')

    log_feedback(query, predicted_output, feedback)

    if feedback == 'no':
        # Preprocess and handle None values
        current_query_vector = vectorizer.transform([preprocess_text(query)])
        global X, df

        # Ensure 'Command' column is string type or categorical
        df['Command'] = df['Command'].astype(str)  # Ensure string type

        # Drop rows where 'Command' is None
        df = df.dropna(subset=['Command'])

        X = vstack([X, current_query_vector])
        df = pd.concat([df, pd.DataFrame({'Command': [predicted_output], 'Description': [query]})], ignore_index=True)

        # Fit the model after cleaning
        model.fit(X, df['Command'])
        joblib.dump(model, model_output_path)
        df.to_json(df_output_path, orient='records', lines=True)
        
        # Simulate time-consuming operation
        time.sleep(5)
        
        print("Model updated based on user feedback.")

    return jsonify({'status': 'feedback received'})


if __name__ == '__main__':
    app.run(debug=True)