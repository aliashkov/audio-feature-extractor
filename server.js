import { Worker } from "worker_threads";
import path from "path";
import Redis from "ioredis";
import { Queue, Worker as BullWorker } from "bullmq";
import { initModels } from "./modelInitializer.js";
import { predict } from "./utils/utils.js";
import { exampleTracks } from "./utils/tracks.js";

const redisConfig = {
  host: process.env.REDIS_HOST || "redis",
  port: process.env.REDIS_PORT || 6379,
  password: process.env.REDIS_PASSWORD,
  maxRetriesPerRequest: 3,
  enableReadyCheck: true,
  retryStrategy: (times) => {
    const delay = Math.min(times * 50, 2000);
    return delay;
  }
};

let isShuttingDown = false;

const createRedisInstance = () => {
  const redis = new Redis(redisConfig);
  
  redis.on('error', (error) => {
    console.error('Redis connection error:', error);
  });

  redis.on('connect', () => {
    console.log('Successfully connected to Redis');
  });

  return redis;
};

const createQueue = (name) => new Queue(name, { 
  connection: redisConfig,
  defaultJobOptions: {
    removeOnComplete: true,
    removeOnFail: true,
    attempts: 3,
    backoff: {
      type: "exponential",
      delay: 60000,
    }
  }
});

const redis = createRedisInstance();

// Initialize BullMQ queues
const inputQueue = createQueue("audio-features");
const outputQueue = createQueue("audio-features-results");

let models = null;
const maxConcurrentWorkers = parseInt(process.env.MAX_CONCURRENT_WORKERS) || 5;

// Add timing tracking
let startTime = null;
let completedJobs = 0;
let totalJobs = 0;

async function loadModels() {
  try {
    models = await initModels();
    console.log("Models initialized and ready to use.");
    return models;
  } catch (error) {
    console.error("Error loading models:", error);
    throw error;
  }
}

async function cleanupDuplicateJobs() {
  const jobs = await inputQueue.getJobs(['active', 'waiting', 'delayed']);
  const seenTrackIds = new Set();
  
  for (const job of jobs) {
    const trackId = job.data.trackId;
    if (seenTrackIds.has(trackId)) {
      console.log(`Removing duplicate job for trackId: ${trackId}`);
      await job.remove();
    } else {
      seenTrackIds.add(trackId);
    }
  }
}

// Create the worker only after models are loaded
async function initializeBullWorker() {
  await loadModels(); // Ensure models are loaded first

  const worker = new BullWorker(
    "audio-features",
    async (job) => {
      if (isShuttingDown) {
        throw new Error('Worker is shutting down');
      }
  
      const jobId = `processed:${job.data.trackId}`;
      const jobStartTime = Date.now();
  
      if (!models) {
        await loadModels();
        if (!models) {
          throw new Error("Models are not initialized yet.");
        }
      }
  
      const { offlineUrl, trackId } = job.data;
      console.log(`Processing job for trackId: ${trackId}, offlineUrl: ${offlineUrl}`);
  
      if (!offlineUrl || offlineUrl.trim() === "") {
        await outputQueue.add("failed", {
          trackId,
          failedReason: "Offline URL is missing or empty",
        }, {
          jobId: `failed:${trackId}`
        });
        throw new Error("Offline URL is missing or empty");
      }

      try {
        const worker = new Worker(path.resolve("worker.js"), {
          workerData: { offlineUrl },
          resourceLimits: {
            maxOldGenerationSizeMb: 512,
            maxYoungGenerationSizeMb: 128,
          },
        });

        return new Promise((resolve, reject) => {
          const timeout = setTimeout(() => {
            worker.terminate();
            outputQueue.add("failed", {
              trackId,
              failedReason: "Worker timeout after 5 minutes",
            }, {
              jobId: `failed:${trackId}`
            });
            reject(new Error("Worker timeout after 5 minutes"));
          }, 5 * 60 * 1000); // 5 minutes

          worker.on("message", async (message) => {
            clearTimeout(timeout);
            if (message.type === "analyze") {
              const predictions = await predict(message.featuresData, models);
              await outputQueue.add("completed", {
                trackId,
                ...predictions,
              }, {
                jobId: `completed:${trackId}`
              });

              await redis.set(jobId, '1', 'EX', 86400); // 24 hours expiry

              completedJobs++;
              const jobDuration = Date.now() - jobStartTime;
              console.log(`Job ${job.id} completed in ${jobDuration}ms`);

              if (completedJobs === totalJobs) {
                const totalDuration = Date.now() - startTime;
                console.log(`\nAll jobs completed! Total execution time: ${totalDuration}ms`);
                console.log(`Average time per job: ${(totalDuration / totalJobs).toFixed(2)}ms`);
              }

              resolve(predictions);
            } else {
              await handleWorkerError(message.error, trackId);
              reject(new Error(message.error));
            }
          });

          worker.on("error", async (error) => {
            await handleWorkerError(error.message, trackId);
            reject(error);
          });

          worker.on("exit", async (code) => {
            clearTimeout(timeout);
            if (code !== 0) {
              await handleWorkerError(`Worker stopped with exit code ${code}`, trackId);
              reject(new Error(`Worker stopped with exit code ${code}`));
            }
          });
        });
      } catch (error) {
        console.error("Processing error:", error);
        await outputQueue.add("failed", {
          trackId,
          failedReason: error.message,
        }, {
          jobId: `failed:${trackId}`
        });
        throw error;
      }
    },
    {
      concurrency: maxConcurrentWorkers,
      connection: redisConfig,
      lockDuration: 300000, // Increased lock duration to 5 minutes
      lockRenewTime: 30000, // 30 seconds
      attempts: 3,
      backoff: {
        type: "exponential",
        delay: 60000,
      },
    }
  );

  worker.on('error', (error) => {
    console.error('Worker error:', error);
  });

  worker.on('failed', (job, error) => {
    console.error(`Job ${job.id} failed:`, error);
  });

  return worker;
}

// Handle worker errors by logging and adding to the failed queue
async function handleWorkerError(errorMessage, trackId) {
  console.error(`Worker error: ${errorMessage}`);
  await outputQueue.add("failed", {
    trackId,
    failedReason: errorMessage,
  }, {
    jobId: `failed:${trackId}`
  });
}

const intervalId = setInterval(() => {
  if (startTime && !isShuttingDown) {
    const elapsedTime = Date.now() - startTime;
    console.log(`Progress: ${completedJobs}/${totalJobs} jobs completed`);
    console.log(`Time elapsed: ${(elapsedTime / 1000).toFixed(2)} seconds`);
  }
}, 30000);

let bullWorker;

// Initialize the worker and then start processing
async function initialize() {
  try {
    await cleanupDuplicateJobs();
    bullWorker = await initializeBullWorker();
    console.log("Worker initialized and ready to process jobs");
  } catch (error) {
    console.error("Failed to initialize worker:", error);
    process.exit(1);
  }
}

// Start the initialization
initialize();

// Function to add jobs to the queue
export async function addJob(data) {
  if (isShuttingDown) {
    throw new Error('Service is shutting down');
  }
  return await inputQueue.add('process-audio', data, {
    jobId: data.trackId,
    removeOnComplete: true,
    removeOnFail: true
  });
}

async function gracefulShutdown() {
  isShuttingDown = true;
  clearInterval(intervalId);

  if (startTime) {
    const totalDuration = Date.now() - startTime;
    console.log(`\nProcess terminated!`);
    console.log(`Completed ${completedJobs}/${totalJobs} jobs`);
    console.log(`Total execution time: ${totalDuration}ms (${(totalDuration / 1000).toFixed(2)} seconds)`);
  }

  try {
    if (bullWorker) {
      console.log('Closing worker...');
      await bullWorker.close(true); // Wait for ongoing jobs
    }
    
    console.log('Closing Redis connection...');
    await redis.quit();
    
    console.log('Shutdown complete');
    process.exit(0);
  } catch (error) {
    console.error('Error during shutdown:', error);
    process.exit(1);
  }
}

// Handle different termination signals
process.on("SIGTERM", gracefulShutdown);
process.on("SIGINT", gracefulShutdown);
process.on("uncaughtException", (error) => {
  console.error('Uncaught Exception:', error);
  gracefulShutdown();
});