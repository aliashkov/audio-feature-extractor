import express from 'express';
import decode from 'audio-decode';
import { predict } from './main.js'

const app = express();

app.post('/predict', (req, res) => {
  const audioUrl = req.body.audioUrl;
  predict(audioUrl).then((predictions) => {
    res.json(predictions);
  }).catch((err) => {
    console.error('Error predicting:', err);
    res.status(500).json({ error: 'Internal Server Error' });
  });
});

app.listen(3000, () => {
  console.log('Server listening on port 3000!');
});
