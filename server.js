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
};

const createRedisInstance = () => new Redis(redisConfig);

const createQueue = (name) => new Queue(name, { connection: redisConfig });

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

  return new BullWorker(
    "audio-features",
    async (job) => {
      // Check if result already exists in output queue
      const completed = await outputQueue.getJobs(['completed']);
      const alreadyCompleted = completed.some(
        completedJob => completedJob.data.trackId === job.data.trackId
      );
      
      if (alreadyCompleted) {
        console.log(`Job ${job.data.trackId} already has results, skipping`);
        return;
      }

      const jobId = `processed:${job.data.trackId}`;
      
      // Check if job was already processed
      const wasProcessed = await redis.get(jobId);
      if (wasProcessed) {
        console.log(`Job ${job.data.trackId} was already processed, skipping`);
        return;
      }

      const jobStartTime = Date.now();

      if (!models) {
        await loadModels(); // Try to reload models if they're not available
        if (!models) {
          throw new Error("Models are not initialized yet.");
        }
      }

      const { offlineUrl, trackId } = job.data;
      console.log(offlineUrl)
      console.log(trackId)

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
          }, 5 * 60 * 1000);

          worker.on("message", async (message) => {
            if (message.type === "analyze") {
              clearTimeout(timeout);
              const predictions = await predict(message.featuresData, models);

              await outputQueue.add("completed", {
                trackId,
                ...predictions,
              }, {
                jobId: `completed:${trackId}`
              });

              // Mark job as processed with 24h expiry
              await redis.set(jobId, '1', 'EX', 86400);

              await worker.terminate();

              completedJobs++;
              const jobDuration = Date.now() - jobStartTime;
              console.log(`Job ${job.id} completed in ${jobDuration}ms`);

              if (completedJobs === totalJobs) {
                const totalDuration = Date.now() - startTime;
                console.log(`\nAll jobs completed!`);
                console.log(
                  `Total execution time: ${totalDuration}ms (${(
                    totalDuration / 1000
                  ).toFixed(2)} seconds)`
                );
                console.log(
                  `Average time per job: ${(totalDuration / totalJobs).toFixed(
                    2
                  )}ms`
                );
              }

              resolve(predictions);
            } else {
              clearTimeout(timeout);
              await worker.terminate();
              await outputQueue.add("failed", {
                trackId,
                failedReason: message.error,
              }, {
                jobId: `failed:${trackId}`
              });
              reject(new Error(message.error));
            }
          });

          worker.on("error", async (error) => {
            clearTimeout(timeout);
            await worker.terminate();
            await outputQueue.add("failed", {
              trackId,
              failedReason: error.message,
            }, {
              jobId: `failed:${trackId}`
            });
            reject(error);
          });

          worker.on("exit", async (code) => {
            clearTimeout(timeout);
            if (code !== 0) {
              await outputQueue.add("failed", {
                trackId,
                failedReason: `Worker stopped with exit code ${code}`,
              }, {
                jobId: `failed:${trackId}`
              });
              reject(new Error(`Worker stopped with exit code ${code}`));
            }
            await worker.terminate();
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
      attempts: 3,
      backoff: {
        type: "exponential",
        delay: 60000,
      },
      defaultJobOptions: {
        removeOnComplete: true,
        removeOnFail: true,
        attempts: 3,
        backoff: {
          type: "exponential",
          delay: 60000,
        },
      },
    }
  );
}

const intervalId = setInterval(() => {
  if (startTime) {
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
  return await inputQueue.add('process-audio', data, {
    jobId: data.trackId,
    removeOnComplete: true,
    removeOnFail: true
  });
}

process.on("SIGTERM", async () => {
  clearInterval(intervalId);

  if (startTime) {
    const totalDuration = Date.now() - startTime;
    console.log(`\nProcess terminated!`);
    console.log(`Completed ${completedJobs}/${totalJobs} jobs`);
    console.log(
      `Total execution time: ${totalDuration}ms (${(
        totalDuration / 1000
      ).toFixed(2)} seconds)`
    );
  }

  if (bullWorker) {
    await bullWorker.close();
  }
  await redis.quit();
  process.exit(0);
});