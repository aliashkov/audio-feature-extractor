import { Worker } from 'worker_threads';
import path from 'path';
import Redis from 'ioredis';
import { Queue, Worker as BullWorker } from 'bullmq';
import { initModels } from './modelInitializer.js';
import { predict } from './utils/utils.js';

const PORT = 3001; // Optional, if you want to log the port
const API_KEY = process.env.API_KEY;

// Redis client setup
const redis = new Redis({
  host: process.env.REDIS_HOST || 'redis',
  port: process.env.REDIS_PORT || 6379,
  password: process.env.REDIS_PASSWORD,
});

// BullMQ Queue setup
const inputQueue = new Queue('audio-features', {
  connection: {
    host: process.env.REDIS_HOST || 'redis',
    port: process.env.REDIS_PORT || 6379,
    password: process.env.REDIS_PASSWORD,
  },
});

const outputQueue = new Queue('audio-features-results', {
  connection: {
    host: process.env.REDIS_HOST || 'redis',
    port: process.env.REDIS_PORT || 6379,
    password: process.env.REDIS_PASSWORD,
  },
});

let models;
const maxConcurrentWorkers = parseInt(process.env.MAX_CONCURRENT_WORKERS) || 5;

// Function to load models
async function loadModels() {
  models = await initModels();
  console.log('Models initialized and ready to use.');
}

// Generate a unique task ID based on the audio URL
function generateTaskId(audioUrl) {
  return `task:${audioUrl}`;
}

// BullMQ Worker for processing audio
const bullWorker = new BullWorker(
  'audio-features',
  async (job) => {
    if (!models) {
      throw new Error('Models are not initialized yet.');
    }

    const { audioUrl } = job.data;
    const taskId = generateTaskId(audioUrl);

    try {
      const worker = new Worker(path.resolve('worker.js'), {
        workerData: { audioUrl },
      });

      return new Promise((resolve, reject) => {
        worker.on('message', async (message) => {
          if (message.type === 'features') {
            const predictions = await predict(message.featuresData, models);

            console.log(predictions)

            // Store result in Redis with 1-hour expiration
            await redis.set(taskId, JSON.stringify(predictions), 'EX', 3600);

            // Add to output queue
            await outputQueue.add('audio-features-results', {
              taskId,
              predictions,
              audioUrl,
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
          // Ensure the worker is terminated properly
          worker.terminate().catch(err => console.error('Worker termination error:', err));
        });
      });
    } catch (error) {
      console.error('Processing error:', error);
      throw error;
    }
  },
  {
    concurrency: maxConcurrentWorkers,
    connection: {
      host: process.env.REDIS_HOST || 'redis',
      port: process.env.REDIS_PORT || 6379,
      password: process.env.REDIS_PASSWORD,
    },
  }
);

// Function to manually add jobs
async function addJobs(audioUrls) {
  if (!Array.isArray(audioUrls)) {
    throw new Error('audioUrls should be an array');
  }

  if (!models) {
    throw new Error('Models are not initialized yet.');
  }

  const jobs = await Promise.all(
    audioUrls.map((audioUrl) =>
      inputQueue.add('audio-features', { audioUrl })
    )
  );

  console.log('Jobs added:', jobs.map((job) => job.id));
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  await bullWorker.close();
  await redis.quit();
  process.exit(0);
});

// Example usage: manually add jobs
const exampleAudioUrls = [
  'https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002a4760-1136-42f5-b75e-7f32f08969ee/002a4760-1136-42f5-b75e-7f32f08969ee.mp3',
  'https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002a20e6-99d4-4037-a792-5c638fd74ac6/002a20e6-99d4-4037-a792-5c638fd74ac6.mp3',
  // Add more URLs if needed
];

// Load models and start processing jobs
loadModels()
  .then(async () => {
    if (models) {
      console.log('Models are ready. Adding example jobs...');
      await addJobs(exampleAudioUrls);
    } else {
      console.error('Failed to initialize models. Exiting...');
      process.exit(1);
    }
  })
  .catch((error) => {
    console.error('Error loading models:', error);
    process.exit(1);
  });