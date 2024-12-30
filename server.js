import express from 'express';
import { Worker } from 'worker_threads';
import path from 'path';
import { initModels } from './modelInitializer.js'; // Ensure this imports your model initialization logic

const app = express();
const PORT = 3001;

// Middleware to parse JSON bodies
app.use(express.json());

let models; // Declare models variable

async function loadModels() {
  models = await initModels(); // Initialize models once
  console.log('Models initialized and ready to use.');
}

// Route to handle predictions
app.post('/predict', (req, res) => {
  const audioUrls = req.body.audioUrls; // Expecting an array of audio URLs

  if (!audioUrls || !Array.isArray(audioUrls)) {
    return res.status(400).json({ error: 'audioUrls are required' });
  }

  const workers = audioUrls.map(audioUrl => {
    return new Promise((resolve, reject) => {
      const worker = new Worker(path.resolve('worker.js'), {
        workerData: { audioUrl }, // Pass audio URL to the worker
      });

      worker.on('message', (predictions) => {
        resolve(predictions);
      });

      worker.on('error', (error) => {
        console.error('Worker error:', error);
        reject(error);
      });

      worker.on('exit', (code) => {
        if (code !== 0) {
          console.error(`Worker stopped with exit code ${code}`);
          reject(new Error('Worker stopped unexpectedly'));
        }
      });
    });
  });

  Promise.all(workers)
    .then(results => {
      res.json(results); // Send all predictions as an array
    })
    .catch(error => {
      res.status(500).json({ error: 'Internal Server Error' });
    });
});

// Start the server and load models
loadModels().then(() => {
  app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}!`);
  });
});