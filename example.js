import fs from 'fs';
import decode from 'audio-decode';
import { Essentia, EssentiaWASM } from 'essentia.js';

// Initialize Essentia
const essentia = new Essentia(EssentiaWASM);

let model;
let modelName = "danceability";
let modelLoaded = false;
let modelReady = false;

const modelTagOrder = {
  'mood_happy': [true, false],
  'mood_sad': [false, true],
  'mood_relaxed': [false, true],
  'mood_aggressive': [true, false],
  'danceability': [true, false]
};

// Use the hosted model URL
const modelPath = 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/danceability-musicnn-msd-2/model.json';

async function loadTensorflowWASM() {
  const tf = await import('@tensorflow/tfjs');
  await tf.ready();
  console.info('tfjs WASM backend successfully initialized!');
  return tf;
}

async function initModel(tf) {
  const { TensorflowMusiCNN } = await import('essentia.js/dist/essentia.js-model.umd.js');
  model = new TensorflowMusiCNN(tf, modelPath);
  await model.initialize();
  console.info(`Model ${modelName} has been loaded!`);
  return true;
}

function getZeroMatrix(x, y) {
  let matrix = new Array(x);
  for (let f = 0; f < x; f++) {
    matrix[f] = new Array(y).fill(0);
  }
  return matrix;
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

const main = async () => {
  try {
    const tf = await loadTensorflowWASM();
    if (!tf) {
      throw new Error('Failed to initialize TensorFlow.js WASM backend');
    }

    modelLoaded = await initModel(tf);

    const buffer = fs.readFileSync('audio/1.mp3');

    const audio = await decode(buffer);

    const data = essentia.arrayToVector(audio._channelData[0]);

    // Extract audio features
    const danceability = essentia.Danceability(data).danceability;
    const duration = essentia.Duration(data).duration;
    const energy = essentia.Energy(data).energy;
    const keyExtractor = essentia.KeyExtractor(data);
    const KEYS = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
    const key = KEYS.indexOf(keyExtractor.key);
    const mode = keyExtractor.scale === 'major' ? 1 : 0;
    const loudness = essentia.DynamicComplexity(data).loudness;
    const tempo = essentia.PercivalBpmEstimator(data).bpm;

    // Warm up the model
    const fakeFeatures = {
      melSpectrum: getZeroMatrix(187, 96),
      frameSize: 187,
      melBandsSize: 96,
      patchSize: 187
    };

    const fakeStart = Date.now();
    const predictions = await model.predict(fakeFeatures, false);
    const summarizedPredictions = twoValuesAverage(predictions);
    console.log('Predictions:', summarizedPredictions);
    console.info(`Model: Warm up inference took: ${Date.now() - fakeStart}`);

  } catch (err) {
    console.error('Error processing audio file:', err);
  }
};

// Run the main function
main();