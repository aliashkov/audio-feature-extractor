import { workerData, parentPort } from 'worker_threads';
import { Essentia, EssentiaWASM, EssentiaModel } from 'essentia.js';
import decode from 'audio-decode';
import fetch from 'node-fetch';
import * as tf from '@tensorflow/tfjs-node';

// Define model paths
const modelPaths = {
  mood_dancability: 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/danceability-musicnn-msd-2/model.json',
  mood_happy: 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/mood_happy-musicnn-msd-2/model.json',
  mood_sad: 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/mood_sad-musicnn-msd-2/model.json',
  mood_relaxed: 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/mood_relaxed-musicnn-msd-2/model.json',
  mood_aggressive: 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/mood_aggressive-musicnn-msd-2/model.json',
};

const essentia = new Essentia(EssentiaWASM);
const extractor = new EssentiaModel.EssentiaTFInputExtractor(EssentiaWASM, 'musicnn', false);

async function initializeModels() {
  const models = {};
  for (const [name, path] of Object.entries(modelPaths)) {
    const { TensorflowMusiCNN } = await import('essentia.js/dist/essentia.js-model.umd.js');
    models[name] = new TensorflowMusiCNN(tf, path);
    await models[name].initialize();
  }
  return models;
}

async function computeFeatures(audioUrl) {
  const response = await fetch(audioUrl);
  const buffer = await response.arrayBuffer();
  const audio = await decode(buffer);
  const data = essentia.arrayToVector(audio._channelData[0]);

  // Extract audio features
  const energy = essentia.Energy(data).energy;
  const loudness = essentia.DynamicComplexity(data).loudness;
  const tempo = essentia.PercivalBpmEstimator(data).bpm;

  const features = await extractor.computeFrameWise(audio._channelData[0], 1024);
  return { features, energy, loudness, tempo };
}




async function predict(featuresData, models) {
  const predictions = {};

  for (const modelName of Object.keys(models)) {
    const selectedModel = models[modelName];
    const predictionsArray = await selectedModel.predict(featuresData.features, true);

    let summarizedPredictions = twoValuesAverage(predictionsArray);
    console.log(summarizedPredictions);
    if (modelName === 'mood_relaxed' || modelName === 'mood_sad') {
      summarizedPredictions = summarizedPredictions.map(value => 1 - value);
    }

    const prediction = summarizedPredictions[0];
    predictions[modelName] = prediction;
  }

  const formattedPredictions = {
    ...predictions,
    energy: featuresData.energy,
    loudness: featuresData.loudness,
    tempo: featuresData.tempo,
  };


  return formattedPredictions;
}

async function run() {
  try {
    // Only compute features in the worker
    const featuresData = await computeFeatures(workerData.audioUrl);

    // console.log(featuresData)
    // Send features back to main thread
    parentPort.postMessage({
      type: 'features',
      featuresData: featuresData
    });
  } catch (error) {
    console.error('Error in worker:', error);
    parentPort.postMessage({
      type: 'error',
      error: 'Error processing audio'
    });
  }
}

run();