import express from 'express';
import { Worker } from 'worker_threads';
import path from 'path';
import { initModels } from './modelInitializer.js'; // Ensure this imports your model initialization logic
import { predict } from "./utils/utils.js"

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
app.post('/predict', async (req, res) => {
  const audioUrls = req.body.audioUrls;
  
  if (!audioUrls || !Array.isArray(audioUrls)) {
      return res.status(400).json({ error: 'audioUrls are required' });
  }
  
  const workers = audioUrls.map(audioUrl => {
      return new Promise((resolve, reject) => {
          const worker = new Worker(path.resolve('worker.js'), {
              workerData: { audioUrl }
          });
          
          worker.on('message', async (message) => {
              try {
                  if (message.type === 'features') {
                      const predictions = await predict(message.featuresData, models);
                      worker.terminate(); // Terminate worker after prediction
                      resolve(predictions);
                  } else if (message.type === 'error') {
                      worker.terminate();
                      reject(new Error(message.error));
                  }
              } catch (error) {
                  worker.terminate();
                  reject(error);
              }
          });
          
          worker.on('error', (error) => {
              console.error('Worker error:', error);
              worker.terminate();
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
  
  try {
      const results = await Promise.all(workers);
      res.json(results);
  } catch (error) {
      console.error('Prediction error:', error);
      res.status(500).json({ error: 'Internal Server Error' });
  }
});




// Start the server and load models
loadModels().then(() => {
  app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}!`);
  });
});