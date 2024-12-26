import fs from 'fs';
import decode from 'audio-decode';
import { Essentia, EssentiaWASM, EssentiaModel } from 'essentia.js';

const KEEP_PERCENTAGE = 0.15; // Keep only 15% of audio file

// Initialize Essentia
const essentia = new Essentia(EssentiaWASM);

const SAMPLE_RATE = 44100;
const CHANNELS = 1;

// Model paths
const modelPaths = {
  mood_happy: 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/mood_happy-musicnn-msd-2/model.json',
  mood_sad: 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/mood_sad-musicnn-msd-2/model.json',
  mood_relaxed: 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/mood_relaxed-musicnn-msd-2/model.json',
  mood_aggressive: 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/mood_aggressive-musicnn-msd-2/model.json',
};

// Extractor initialization
const extractor = new EssentiaModel.EssentiaTFInputExtractor(EssentiaWASM, 'musicnn', false);

async function loadTensorflowWASM() {
  const tf = await import('@tensorflow/tfjs-node');
  await tf.ready();
  console.info('tfjs WASM backend successfully initialized!');
  return tf;
}

async function decodeAudioWebAPI(filepath) {
  try {
      const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
      const buffer = fs.readFileSync(filepath);
      const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      return audioBuffer.getChannelData(0); // Returns Float32Array
  } catch (error) {
      console.error('Error decoding audio with Web Audio API:', error);
      throw error;
  }
}

async function initModels(tf) {
  const models = {};
  for (const [name, path] of Object.entries(modelPaths)) {
    const { TensorflowMusiCNN } = await import('essentia.js/dist/essentia.js-model.umd.js');
    models[name] = new TensorflowMusiCNN(tf, path);
    await models[name].initialize();
    console.info(`Model ${name} has been loaded!`);
  }
  return models; // Return models object
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

    const models = await initModels(tf); // Store models in an object

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

    console.log(`Danceability: ${danceability}, Duration: ${duration}, Energy: ${energy}, Key: ${key}, Mode: ${mode}, Loudness: ${loudness}, Tempo: ${tempo}`);

    // Prepare features for prediction
    const features = extractor.computeFrameWise(audio._channelData[0], 256); // Adjust frame size here if needed

    // Predict using all models
    for (const modelName of Object.keys(models)) {
      const selectedModel = models[modelName]; // Use the current model
      const predictions = await selectedModel.predict(features, true);
      const summarizedPredictions = twoValuesAverage(predictions);
      console.log(`Predictions for ${modelName}:`, summarizedPredictions);
    }

  } catch (err) {
    console.error('Error processing audio file:', err);
  }
};

function shortenAudio(audioIn, keepRatio = 0.5, trim = false) {
  if (keepRatio < 0.15) {
    keepRatio = 0.15; // Must keep at least 15% of the file
  } else if (keepRatio > 0.66) {
    keepRatio = 0.66; // Will keep at most 2/3 of the file
  }

  if (trim) {
    const discardSamples = Math.floor(0.1 * audioIn.length); // Discard 10% on beginning and end
    audioIn = audioIn.subarray(discardSamples, audioIn.length - discardSamples); // Create new view of buffer without beginning and end
  }

  const ratioSampleLength = Math.ceil(audioIn.length * keepRatio);
  const patchSampleLength = 187 * 256; // Cut into patchSize chunks so there's no weird jumps in audio
  const numPatchesToKeep = Math.ceil(ratioSampleLength / patchSampleLength);

  // Space patchesToKeep evenly
  const skipSize = Math.floor((audioIn.length - ratioSampleLength) / (numPatchesToKeep - 1));

  let audioOut = [];
  let startIndex = 0;
  for (let i = 0; i < numPatchesToKeep; i++) {
    let endIndex = startIndex + patchSampleLength;
    let chunk = audioIn.slice(startIndex, endIndex);
    audioOut.push(...chunk);
    startIndex = endIndex + skipSize; // Discard even space
  }

  return Float32Array.from(audioOut);
}

// Run the main function
main();
