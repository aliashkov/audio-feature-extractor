import { workerData, parentPort } from 'worker_threads';
import { Essentia, EssentiaWASM, EssentiaModel } from 'essentia.js';
import decode from 'audio-decode';
import fetch from 'node-fetch';
import * as tf from '@tensorflow/tfjs-node';

const essentia = new Essentia(EssentiaWASM);
const extractor = new EssentiaModel.EssentiaTFInputExtractor(EssentiaWASM, 'musicnn', false);

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
    const features = await computeFeatures(workerData.audioUrl);
    const predictions = await predict(features, workerData.models);
    parentPort.postMessage(predictions);
  } catch (error) {
    console.error('Error in worker:', error);
    parentPort.postMessage({ error: 'Error processing audio' });
  }
}

run();