# Documentation: Installation and Setup

### 1. Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.7+
- pip (Python package installer)
- Node.js and npm (Node package manager)

### 2. Directory Structure

Ensure your project directory is structured as follows:

```go
goCopy code
adapt_model_backend/
├── model_backend/
│   ├── app.py
│   ├── nmap_command_classifier.pkl
│   ├── nmap_commands_updated.json
│   ├── feedback_log.txt
│   ├── requirements.txt
│   ├── venv/
│   └── __pycache__/
├── node_backend/
│   ├── app.js
│   ├── package.json
│   ├── package-lock.json
│   └── node_modules/
└── README.md

```

### 3. Setting up the Flask Backend

### Step 1: Create a Virtual Environment

Navigate to the `model_backend` directory and create a virtual environment:

```bash
bashCopy code
cd adapt_model_backend/model_backend
python -m venv venv

```

Activate the virtual environment:

- On Windows:
    
    ```bash
    bashCopy code
    venv\Scripts\activate
    
    ```
    
- On macOS/Linux:
    
    ```bash
    bashCopy code
    source venv/bin/activate
    
    ```
    

### Step 2: Install Python Dependencies

Create a `requirements.txt` file with the following content:

```
Copy code
flask
flask-cors
nltk
scikit-learn
joblib
pandas

```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data

In your `app.py`, ensure you have the following NLTK downloads:

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

```

### Step 4: Create and Configure `app.py`

Here’s the complete `app.py` file:

```python
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
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return ' '.join(cleaned_tokens)

df['Processed_Description'] = df['Description'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Processed_Description'])

@app.route('/predict', methods=['POST'])
def predict():
    query = request.json.get('query')
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])
    predicted_command = model.predict(query_vector)
    return jsonify({'command': predicted_command[0]})

@app.route('/feedback', methods=['POST'])
def feedback():
    query = request.json.get('query')
    predicted_output = request.json.get('predicted_output')
    feedback = request.json.get('feedback')

    with open("feedback_log.txt", "a") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Predicted Output: {predicted_output}\n")
        f.write(f"Feedback: {feedback}\n")
        f.write("="*30 + "\n")

    if feedback == 'no':
        current_query = preprocess_text(query)
        df = df.append({'Command': predicted_output, 'Description': current_query}, ignore_index=True)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['Description'].apply(preprocess_text))
        model.fit(X, df['Command'])
        joblib.dump(model, model_output_path)
        df.to_json(df_output_path, orient='records', lines=True)

    return jsonify({'status': 'feedback logged and model updated'})

if __name__ == '__main__':
    app.run(debug=True)

```

### 4. Setting up the Express Backend

### Step 1: Navigate to the `node_backend` Directory

```bash

cd ../node_backend
```

### Step 2: Initialize a Node.js Project

Run the following commands to initialize the project and install required packages:

```bash
npm init -y
npm install express axios cors
```

### Step 3: Create `app.js`

Here’s the complete `app.js` file:

```jsx
const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

const flaskBackendUrl = 'http://localhost:5000';

app.post('/predict', async (req, res) => {
    try {
        const response = await axios.post(`${flaskBackendUrl}/predict`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).send(error.toString());
    }
});

app.post('/feedback', async (req, res) => {
    try {
        const response = await axios.post(`${flaskBackendUrl}/feedback`, req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).send(error.toString());
    }
});

const port = 3000;
app.listen(port, () => {
    console.log(`Node.js server running on port ${port}`);
});

```

### 5. Running the Project

### Step 1: Run the Flask Backend

Navigate to `model_backend` directory and activate the virtual environment:

```bash
cd ../model_backend
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux

```

Run the Flask app:

```bash
python app.py
```

### Step 2: Run the Node.js Backend

Open another terminal window, navigate to the `node_backend` directory, and start the server:

```bash
cd ../node_backend
node app.js
```

### 6. Testing the Endpoints

Use tools like Postman or `curl` to test the endpoints.

### Predict Endpoint

```bash
curl -X POST http://localhost:3000/predict -H "Content-Type: application/json" -d "{\"query\": \"port scan\"}"
```

### Feedback Endpoint

```bash
curl -X POST http://localhost:3000/feedback -H "Content-Type: application/json" -d "{\"query\": \"port scan\", \"predicted_output\": \"-sS\", \"feedback\": \"no\"}"
```

### 7. Conclusion

Your Flask and Node.js backend setup should now be up and running, ready to handle requests and provide predictions and feedback mechanisms.

If you encounter any issues, ensure all dependencies are correctly installed, and that both servers are running without errors.