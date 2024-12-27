import express from 'express';
import decode from 'audio-decode';
import { predict } from './main.js';

const app = express();

// Middleware to parse JSON bodies
app.use(express.json());

app.post('/predict', (req, res) => {
  console.log(77777);
  const audioUrl = req.body.audioUrl; // Ensure audioUrl is coming from the request body
  console.log(audioUrl);

  if (!audioUrl) {
    return res.status(400).json({ error: 'audioUrl is required' });
  }

  predict(audioUrl)
    .then((predictions) => {
      res.json(predictions);
    })
    .catch((err) => {
      console.error('Error predicting:', err);
      res.status(500).json({ error: 'Internal Server Error' });
    });
});

app.listen(3001, () => {
  console.log('Server listening on port 3001!');
});
