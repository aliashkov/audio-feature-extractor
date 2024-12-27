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
  
  const features = await extractor.computeFrameWise(audio._channelData[0], 256);
  return features;
}

function twoValuesAverage(arrayOfArrays) {
  let firstValues = [];
  let secondValues = [];

  arrayOfArrays.forEach((v) => {
    firstValues.push(v[0]);
    secondValues.push(v[1]);
  });

  const firstValuesAvg = firstValues.reduce((acc, val) => acc + val) / firstValues.length;
  const secondValuesAvg = secondValues.reduce((acc, val) => acc + val) / secondValues.length;

  return [firstValuesAvg, secondValuesAvg];
}

async function predict(features, models) {
  const predictions = {};
  
  for (const modelName of Object.keys(models)) {
    const selectedModel = models[modelName];
    const predictionsArray = await selectedModel.predict(features, true);

    let summarizedPredictions = twoValuesAverage(predictionsArray);
    if (modelName === 'mood_relaxed' || modelName === 'mood_sad') {
      summarizedPredictions = summarizedPredictions.map(value => 1 - value);
    }

    predictions[modelName] = summarizedPredictions;
  }

  return predictions;
}

async function run() {
  try {
    const models = await initializeModels(); // Initialize models here
    const features = await computeFeatures(workerData.audioUrl);
    const predictions = await predict(features, models);
    parentPort.postMessage(predictions);
  } catch (error) {
    console.error('Error in worker:', error);
    parentPort.postMessage({ error: 'Error processing audio' });
  }
}

run();