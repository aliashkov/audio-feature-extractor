import express from 'express';
import { Worker } from 'worker_threads';
import path from 'path';
import Redis from 'ioredis';
import { initModels } from './modelInitializer.js';
import { predict } from "./utils/utils.js";

const app = express();
const PORT = 3001;
const API_KEY = process.env.API_KEY;

// Redis client setup
const redis = new Redis({
  host: process.env.REDIS_HOST || 'redis',
  port: process.env.REDIS_PORT || 6379,
  password: process.env.REDIS_PASSWORD,
});

app.use(express.json());

// Authentication middleware
const authenticateApiKey = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1]; // Extract the token from the Bearer format
  
    console.log(token); // Log the token to check if it's being retrieved
  
    if (!token || token !== API_KEY) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
  
    next();
};

let models;
const maxConcurrentWorkers = parseInt(process.env.MAX_CONCURRENT_WORKERS) || 5;
const queue = [];
let activeWorkers = 0;

async function loadModels() {
  models = await initModels();
  console.log('Models initialized and ready to use.');
}

// Generate a unique task ID based on audio URL
function generateTaskId(audioUrl) {
  return `task:${audioUrl}`;
}

async function processNext() {
  if (queue.length === 0 || activeWorkers >= maxConcurrentWorkers) {
    return;
  }

  const { audioUrl, taskId, resolve, reject } = queue.shift();
  
  // Check if task is already being processed
  const existingResult = await redis.get(taskId);
  if (existingResult) {
    resolve(JSON.parse(existingResult));
    processNext();
    return;
  }

  activeWorkers++;

  const worker = new Worker(path.resolve('worker.js'), {
    workerData: { audioUrl }
  });

  worker.on('message', async (message) => {
    if (message.type === 'features') {
      const predictions = await predict(message.featuresData, models);
      // Store result in Redis with 1 hour expiration
      await redis.set(taskId, JSON.stringify(predictions), 'EX', 3600);
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

app.get('/health', authenticateApiKey, (req, res) => {
  const healthStatus = {
    status: 'UP',
    modelsInitialized: !!models,
  };
  
  res.json(healthStatus);
});

app.post('/predict', authenticateApiKey, (req, res) => {
  const audioUrls = req.body.audioUrls;

  if (!audioUrls || !Array.isArray(audioUrls)) {
    return res.status(400).json({ error: 'audioUrls are required' });
  }

  const promises = audioUrls.map(audioUrl => {
    const taskId = generateTaskId(audioUrl);
    
    return new Promise((resolve, reject) => {
      queue.push({ audioUrl, taskId, resolve, reject });
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

// Graceful shutdown
process.on('SIGTERM', async () => {
  await redis.quit();
  process.exit(0);
});

loadModels().then(() => {
  app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}!`);
  });
});