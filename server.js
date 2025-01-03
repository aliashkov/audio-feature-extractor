import express from 'express';
import { Worker } from 'worker_threads';
import path from 'path';
import { initModels } from './modelInitializer.js'; // Ensure this imports your model initialization logic
import { predict } from "./utils/utils.js";

const app = express();
const PORT = 3001;

app.use(express.json());

let models; 
const maxConcurrentWorkers = 5;
const queue = [];
let activeWorkers = 0;

async function loadModels() {
  models = await initModels(); // Initialize models once
  console.log('Models initialized and ready to use.');
}

function processNext() {
  if (queue.length === 0 || activeWorkers >= maxConcurrentWorkers) {
    return; // No tasks to process or limit reached
  }

  const { audioUrl, resolve, reject } = queue.shift();
  activeWorkers++;

  const worker = new Worker(path.resolve('worker.js'), {
    workerData: { audioUrl }
  });

  worker.on('message', async (message) => {
    if (message.type === 'features') {
      const predictions = await predict(message.featuresData, models);
      resolve(predictions);
    } else {
      reject(new Error(message.error));
    }
    activeWorkers--;
    processNext();
  });

  worker.on('error', (error) => {
    console.error('Worker error:', error);
    activeWorkers--;
    reject(error);
    processNext();
  });

  worker.on('exit', (code) => {
    if (code !== 0) {
      console.error(`Worker stopped with exit code ${code}`);
      reject(new Error('Worker stopped unexpectedly'));
    }
  });
}

// Route to handle predictions
app.post('/predict', (req, res) => {
  const audioUrls = req.body.audioUrls;

  if (!audioUrls || !Array.isArray(audioUrls)) {
    return res.status(400).json({ error: 'audioUrls are required' });
  }

  const promises = audioUrls.map(audioUrl => {
    return new Promise((resolve, reject) => {
      queue.push({ audioUrl, resolve, reject });
      processNext();
    });
  });

  Promise.all(promises)
    .then(results => res.json(results))
    .catch(error => {
      console.error('Prediction error:', error);
      res.status(500).json({ error: 'Internal Server Error' });
    });
});

loadModels().then(() => {
  app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}!`);
  });
});