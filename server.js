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
const maxConcurrentWorkers = parseInt(process.env.MAX_CONCURRENT_WORKERS) || 3;

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
            // await redis.set(taskId, JSON.stringify(predictions), 'EX', 3600);

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

const exampleAudioUrls = [
  'https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002a4760-1136-42f5-b75e-7f32f08969ee/002a4760-1136-42f5-b75e-7f32f08969ee.mp3',
  'https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002a20e6-99d4-4037-a792-5c638fd74ac6/002a20e6-99d4-4037-a792-5c638fd74ac6.mp3',
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/00277e04-faab-4dda-8524-5c7cc11ad3b4/00277e04-faab-4dda-8524-5c7cc11ad3b4.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/00279b78-6acb-4807-85da-0fae325caeaf/00279b78-6acb-4807-85da-0fae325caeaf.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/00287ae5-b072-41a4-b865-9d518da0e4cc/00287ae5-b072-41a4-b865-9d518da0e4cc.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/00288e32-ee23-401f-acc1-2ad44b1b1a93/00288e32-ee23-401f-acc1-2ad44b1b1a93.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/0028cae1-7a66-4c9c-8c7c-4eb61ef304d8/0028cae1-7a66-4c9c-8c7c-4eb61ef304d8.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/00292d78-3625-4690-b6f7-75b6ad7c41c4/00292d78-3625-4690-b6f7-75b6ad7c41c4.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002947d7-d2dd-4979-bb4a-23e5e496ec56/002947d7-d2dd-4979-bb4a-23e5e496ec56.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/0029e583-4369-4831-823d-24dd4f7b7bed/0029e583-4369-4831-823d-24dd4f7b7bed.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/0029fdec-d126-40fe-8277-fb22fdcf0c3a/0029fdec-d126-40fe-8277-fb22fdcf0c3a.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002a20e6-99d4-4037-a792-5c638fd74ac6/002a20e6-99d4-4037-a792-5c638fd74ac6.mp3",
  
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