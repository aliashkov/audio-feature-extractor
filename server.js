import express from 'express';
import { Worker } from 'worker_threads';
import path from 'path';
import Redis from 'ioredis';
import { Queue, Worker as BullWorker } from 'bullmq';
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

// BullMQ Queue setup
const inputQueue = new Queue('audioProcessing', {
  connection: {
    host: process.env.REDIS_HOST || 'redis',
    port: process.env.REDIS_PORT || 6379,
    password: process.env.REDIS_PASSWORD,
  }
});

const outputQueue = new Queue('processedResults', {
  connection: {
    host: process.env.REDIS_HOST || 'redis',
    port: process.env.REDIS_PORT || 6379,
    password: process.env.REDIS_PASSWORD,
  }
});

app.use(express.json());

const authenticateApiKey = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    
    if (!token || token !== API_KEY) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
    next();
};

let models;
const maxConcurrentWorkers = parseInt(process.env.MAX_CONCURRENT_WORKERS) || 5;

async function loadModels() {
  models = await initModels();
  console.log('Models initialized and ready to use.');
}

function generateTaskId(audioUrl) {
  return `task:${audioUrl}`;
}

// BullMQ Worker for processing audio
const bullWorker = new BullWorker('audioProcessing', async job => {
  const { audioUrl } = job.data;
  const taskId = generateTaskId(audioUrl);

  try {
    const worker = new Worker(path.resolve('worker.js'), {
      workerData: { audioUrl }
    });

    return new Promise((resolve, reject) => {
      worker.on('message', async (message) => {
        if (message.type === 'features') {
          const predictions = await predict(message.featuresData, models);
          // Store result in Redis with 1 hour expiration
          await redis.set(taskId, JSON.stringify(predictions), 'EX', 3600);
          // Add to output queue
          await outputQueue.add('processedResult', {
            taskId,
            predictions,
            audioUrl
          });
          resolve(predictions);
        } else {
          reject(new Error(message.error));
        }
      });

      worker.on('error', reject);
      worker.on('exit', (code) => {
        if (code !== 0) {
          reject(new Error(`Worker stopped with exit code ${code}`));
        }
      });
    });
  } catch (error) {
    console.error('Processing error:', error);
    throw error;
  }
}, {
  concurrency: maxConcurrentWorkers,
  connection: {
    host: process.env.REDIS_HOST || 'redis',
    port: process.env.REDIS_PORT || 6379,
    password: process.env.REDIS_PASSWORD,
  }
});

app.get('/health', authenticateApiKey, (req, res) => {
  const healthStatus = {
    status: 'UP',
    modelsInitialized: !!models,
  };
  
  res.json(healthStatus);
});

app.post('/predict', authenticateApiKey, async (req, res) => {
  const audioUrls = req.body.audioUrls;

  if (!audioUrls || !Array.isArray(audioUrls)) {
    return res.status(400).json({ error: 'audioUrls are required' });
  }

  try {
    const jobs = await Promise.all(
      audioUrls.map(audioUrl => 
        inputQueue.add('processAudio', { audioUrl })
      )
    );

    // Return job IDs for tracking
    res.json({
      success: true,
      jobIds: jobs.map(job => job.id)
    });
  } catch (error) {
    console.error('Queue error:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// Endpoint to check job status
app.get('/status/:jobId', authenticateApiKey, async (req, res) => {
  try {
    const job = await inputQueue.getJob(req.params.jobId);
    if (!job) {
      return res.status(404).json({ error: 'Job not found' });
    }

    const state = await job.getState();
    const result = job.returnvalue;

    res.json({
      jobId: job.id,
      state,
      result: result || null
    });
  } catch (error) {
    console.error('Status check error:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  await bullWorker.close();
  await redis.quit();
  process.exit(0);
});

loadModels().then(() => {
  app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}!`);
  });
});