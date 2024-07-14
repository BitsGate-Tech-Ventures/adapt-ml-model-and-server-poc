// server.js

const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');

const app = express();
const port = 3000;

app.use(bodyParser.json());

app.get('/', (req, res) => {
    res.send('Nmap Command Predictor API');
});

// Endpoint for predicting Nmap commands
app.post('/predict', async (req, res) => {
    const { query } = req.body;

    try {
        // Replace with your actual Python backend URL
        const response = await axios.post('http://localhost:5000/predict', { query });
        res.json(response.data);
    } catch (error) {
        res.status(500).send('Error predicting command');
    }
});

// Endpoint for receiving feedback
app.post('/feedback', async (req, res) => {
    const { query, predicted_output, feedback } = req.body;
    console.log(query, predicted_output, feedback);

    try {
        // Replace with your actual Python backend URL
        const response = await axios.post('http://localhost:5000/feedback', { query, predicted_output, feedback });
        res.json(response.data);
    } catch (error) {
        res.status(500).send('Error sending feedback');
    }
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
