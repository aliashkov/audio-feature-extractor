import { Worker } from 'worker_threads';
import path from 'path';
import Redis from 'ioredis';
import { Queue, Worker as BullWorker } from 'bullmq';
import { initModels } from './modelInitializer.js';
import { predict } from './utils/utils.js';
import { exampleTracks } from './utils/tracks.js';


const redisConfig = {
  host: process.env.REDIS_HOST || 'redis',
  port: process.env.REDIS_PORT || 6379,
  password: process.env.REDIS_PASSWORD,
};

const createRedisInstance = () => new Redis(redisConfig);

const createQueue = (name) => new Queue(name, { connection: redisConfig });

const redis = createRedisInstance();

// Initialize BullMQ queues
const inputQueue = createQueue('audio-features');
const outputQueue = createQueue('audio-features-results');

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

const bullWorker = new BullWorker(
  'audio-features',
  async (job) => {
    const jobStartTime = Date.now();

    if (!models) {
      throw new Error('Models are not initialized yet.');
    }

    const { offlineUrl, trackId } = job.data;

    try {
      const worker = new Worker(path.resolve('worker.js'), {
        workerData: { offlineUrl },
        resourceLimits: {
          maxOldGenerationSizeMb: 512,
          maxYoungGenerationSizeMb: 128,
        }
      });

      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          worker.terminate();
          outputQueue.add('failed', {
            trackId,
            failedReason: 'Worker timeout after 5 minutes'
          });
          reject(new Error('Worker timeout after 5 minutes'));
        }, 5 * 60 * 1000);

        worker.on('message', async (message) => {
          console.log(message.type)
          if (message.type === 'analyze') {
            clearTimeout(timeout);
            const predictions = await predict(message.featuresData, models);

            await outputQueue.add('completed', {
              trackId,
              ...predictions,
            });

            await worker.terminate();

            // Track job completion
            completedJobs++;
            const jobDuration = Date.now() - jobStartTime;
            console.log(`Job ${job.id} completed in ${jobDuration}ms`);

            if (completedJobs === totalJobs) {
              const totalDuration = Date.now() - startTime;
              console.log(`\nAll jobs completed!`);
              console.log(`Total execution time: ${totalDuration}ms (${(totalDuration / 1000).toFixed(2)} seconds)`);
              console.log(`Average time per job: ${(totalDuration / totalJobs).toFixed(2)}ms`);
            }

            resolve(predictions);
          } else {
            clearTimeout(timeout);
            await worker.terminate();
            await outputQueue.add('failed', {
              trackId,
              failedReason: message.error
            });
            reject(new Error(message.error));
          }
        });

        worker.on('error', async (error) => {
          clearTimeout(timeout);
          await worker.terminate();
          await outputQueue.add('failed', {
            trackId,
            failedReason: error.message
          });
          reject(error);
        });

        worker.on('exit', async (code) => {
          clearTimeout(timeout);
          if (code !== 0) {
            await outputQueue.add('failed', {
              trackId,
              failedReason: `Worker stopped with exit code ${code}`
            });
            reject(new Error(`Worker stopped with exit code ${code}`));
          }
          await worker.terminate();
        });
      });
    } catch (error) {
      console.error('Processing error:', error);
      await outputQueue.add('failed', {
        trackId,
        failedReason: error.message
      });
      throw error;
    }
  },
  {
    concurrency: maxConcurrentWorkers,
    connection: redisConfig,
  }
);

const intervalId = setInterval(() => {
  // Log progress
  if (startTime) {
    const elapsedTime = Date.now() - startTime;
    console.log(`Progress: ${completedJobs}/${totalJobs} jobs completed`);
    console.log(`Time elapsed: ${(elapsedTime / 1000).toFixed(2)} seconds`);
  }
}, 30000);

async function addJobs(tracks) {
  if (!Array.isArray(tracks)) {
    throw new Error('Tracks should be an array');
  }

  if (!models) {
    throw new Error('Models are not initialized yet.');
  }

  // Initialize timing tracking
  startTime = Date.now();
  completedJobs = 0;
  totalJobs = tracks.length;

  console.log(`Starting processing of ${totalJobs} jobs at ${new Date().toISOString()}`);

  const jobs = await Promise.all(
    tracks.map(({ trackId, offlineUrl }) =>
      
      inputQueue.add(
        'audio-features',
        { trackId, offlineUrl },
        {
          removeOnComplete: true,
          removeOnFail: true,
        }
      )
    )
  );

  console.log('Jobs added:', jobs.map((job) => job.id));
}



loadModels()
  .then(async () => {
    if (models) {
      console.log('Models are ready. Adding the first batch of jobs...');
/* 
      // Add only the first 5 tracks to the queue
      const initialBatch = exampleTracks.slice(0, 5);
      await addJobs(initialBatch);

      console.log('First batch of 5 jobs added.');

      // Optional: Add logic to process the remaining tracks later
      const remainingTracks = exampleTracks.slice(5);
      if (remainingTracks.length > 0) {
        console.log(`There are ${remainingTracks.length} remaining tracks to process.`);
        // Add more jobs as needed, e.g., after some delay
        setTimeout(async () => {
          console.log('Adding remaining jobs...');
          await addJobs(remainingTracks);
        }, 60000); // Add remaining jobs after 60 seconds
      } */
    } else {
      console.error('Failed to initialize models. Exiting...');
      process.exit(1);
    }
  })
  .catch((error) => {
    console.error('Error loading models:', error);
    process.exit(1);
  });


process.on('SIGTERM', async () => {
  clearInterval(intervalId);

  // Log final statistics if process is terminated
  if (startTime) {
    const totalDuration = Date.now() - startTime;
    console.log(`\nProcess terminated!`);
    console.log(`Completed ${completedJobs}/${totalJobs} jobs`);
    console.log(`Total execution time: ${totalDuration}ms (${(totalDuration / 1000).toFixed(2)} seconds)`);
  }

  await bullWorker.close();
  await redis.quit();
  process.exit(0);
});