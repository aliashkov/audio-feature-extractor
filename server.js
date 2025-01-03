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

// Add timing tracking
let startTime = null;
let completedJobs = 0;
let totalJobs = 0;

async function loadModels() {
  models = await initModels();
  console.log('Models initialized and ready to use.');
}

function generateTaskId(audioUrl) {
  return `task:${audioUrl}`;
}

const bullWorker = new BullWorker(
  'audio-features',
  async (job) => {
    const jobStartTime = Date.now();
    
    if (!models) {
      throw new Error('Models are not initialized yet.');
    }

    const { audioUrl } = job.data;
    const taskId = generateTaskId(audioUrl);

    try {
      const worker = new Worker(path.resolve('worker.js'), {
        workerData: { audioUrl },
        resourceLimits: {
          maxOldGenerationSizeMb: 512,
          maxYoungGenerationSizeMb: 128,
        }
      });

      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          worker.terminate();
          reject(new Error('Worker timeout after 5 minutes'));
        }, 5 * 60 * 1000);

        worker.on('message', async (message) => {
          if (message.type === 'features') {
            clearTimeout(timeout);
            const predictions = await predict(message.featuresData, models);

            await outputQueue.add('audio-features-results', {
              taskId,
              predictions,
              audioUrl,
            });

            await worker.terminate();
            
            // Track job completion
            completedJobs++;
            const jobDuration = Date.now() - jobStartTime;
            console.log(`Job ${job.id} completed in ${jobDuration}ms`);
            
            if (completedJobs === totalJobs) {
              const totalDuration = Date.now() - startTime;
              console.log(`\nAll jobs completed!`);
              console.log(`Total execution time: ${totalDuration}ms (${(totalDuration/1000).toFixed(2)} seconds)`);
              console.log(`Average time per job: ${(totalDuration/totalJobs).toFixed(2)}ms`);
            }
            
            resolve(predictions);
          } else {
            clearTimeout(timeout);
            await worker.terminate();
            reject(new Error(message.error));
          }
        });

        worker.on('error', async (error) => {
          clearTimeout(timeout);
          await worker.terminate();
          reject(error);
        });

        worker.on('exit', async (code) => {
          clearTimeout(timeout);
          if (code !== 0) {
            reject(new Error(`Worker stopped with exit code ${code}`));
          }
          await worker.terminate();
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

const used = process.memoryUsage();
const intervalId = setInterval(() => {
  console.log('Memory usage:', {
    rss: `${Math.round(used.rss / 1024 / 1024)}MB`,
    heapTotal: `${Math.round(used.heapTotal / 1024 / 1024)}MB`,
    heapUsed: `${Math.round(used.heapUsed / 1024 / 1024)}MB`,
    external: `${Math.round(used.external / 1024 / 1024)}MB`,
  });
  
  // Log progress
  if (startTime) {
    const elapsedTime = Date.now() - startTime;
    console.log(`Progress: ${completedJobs}/${totalJobs} jobs completed`);
    console.log(`Time elapsed: ${(elapsedTime/1000).toFixed(2)} seconds`);
  }
}, 30000);

async function addJobs(audioUrls) {
  if (!Array.isArray(audioUrls)) {
    throw new Error('audioUrls should be an array');
  }

  if (!models) {
    throw new Error('Models are not initialized yet.');
  }

  // Initialize timing tracking
  startTime = Date.now();
  completedJobs = 0;
  totalJobs = audioUrls.length;
  
  console.log(`Starting processing of ${totalJobs} jobs at ${new Date().toISOString()}`);

  const jobs = await Promise.all(
    audioUrls.map((audioUrl) =>
      inputQueue.add('audio-features', { audioUrl }, {
        removeOnComplete: true,
        removeOnFail: true
      })
    )
  );

  console.log('Jobs added:', jobs.map((job) => job.id));
}

process.on('SIGTERM', async () => {
  clearInterval(intervalId);
  
  // Log final statistics if process is terminated
  if (startTime) {
    const totalDuration = Date.now() - startTime;
    console.log(`\nProcess terminated!`);
    console.log(`Completed ${completedJobs}/${totalJobs} jobs`);
    console.log(`Total execution time: ${totalDuration}ms (${(totalDuration/1000).toFixed(2)} seconds)`);
  }
  
  await bullWorker.close();
  await redis.quit();
  process.exit(0);
});

const exampleAudioUrls = [
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002b7fb1-42f5-4e88-a2cf-7f87b2dff7a9/002b7fb1-42f5-4e88-a2cf-7f87b2dff7a9.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002a4760-1136-42f5-b75e-7f32f08969ee/002a4760-1136-42f5-b75e-7f32f08969ee.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/0029fdec-d126-40fe-8277-fb22fdcf0c3a/0029fdec-d126-40fe-8277-fb22fdcf0c3a.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002d2c1e-e483-4db2-84aa-7b5cf8c5f614/002d2c1e-e483-4db2-84aa-7b5cf8c5f614.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002cad9b-932b-48c0-b842-470e222c939a/002cad9b-932b-48c0-b842-470e222c939a.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002c2360-cb7e-4281-a73e-95e09b80cd60/002c2360-cb7e-4281-a73e-95e09b80cd60.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002bd740-635c-4535-9974-bac90bc574d2/002bd740-635c-4535-9974-bac90bc574d2.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002bc3c3-f410-400a-ac5f-0605c1f9c517/002bc3c3-f410-400a-ac5f-0605c1f9c517.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002bc0a8-b53b-4f20-8e49-cef2d9e5a1a3/002bc0a8-b53b-4f20-8e49-cef2d9e5a1a3.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002b7fb1-42f5-4e88-a2cf-7f87b2dff7a9/002b7fb1-42f5-4e88-a2cf-7f87b2dff7a9.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002b667c-a888-42d5-8ef4-487d8f0495f5/002b667c-a888-42d5-8ef4-487d8f0495f5.mp3",
  "https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002ad4c0-96d8-4eb1-827c-323a8192c045/002ad4c0-96d8-4eb1-827c-323a8192c045.mp3"

];

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