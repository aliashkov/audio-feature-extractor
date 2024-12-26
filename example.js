import fs from 'fs';
import decode from 'audio-decode';
import * as tf from '@tensorflow/tfjs-node';
import { Essentia, EssentiaWASM } from 'essentia.js';
import { TensorflowMusiCNN } from 'essentia.js/dist/essentia.js-model.umd.js';

// Initialize Essentia
const essentia = new Essentia(EssentiaWASM);

let modelName = "";

const modelTagOrder = {
  'mood_happy': [true, false],
  'mood_sad': [false, true],
  'mood_relaxed': [false, true],
  'mood_aggressive': [true, false],
  'danceability': [true, false]
};

// Use the hosted model URL
const modelPath = 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/danceability-musicnn-msd-2/model.json';

const main = async () => {
  try {
    // Load the MP3 file into a buffer
    const buffer = fs.readFileSync('audio/2.mp3');

    // Decode the MP3 file into raw PCM audio data
    const audio = await decode(buffer);

    // Convert the first audio channel to an Essentia vector
    const data = essentia.arrayToVector(audio._channelData[0]);

    // Extract audio features
    const danceability = essentia.Danceability(data).danceability;
    const duration = essentia.Duration(data).duration;
    const energy = essentia.Energy(data).energy;

    // Key extraction
    const keyExtractor = essentia.KeyExtractor(data);
    const KEYS = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
    const key = KEYS.indexOf(keyExtractor.key);
    const mode = keyExtractor.scale === 'major' ? 1 : 0;

    // Loudness extraction
    const loudness = essentia.DynamicComplexity(data).loudness;

    // Tempo extraction
    const tempo = essentia.PercivalBpmEstimator(data).bpm;

    // Initialize the TensorFlow MusicCNN model
    const musiCNN = new TensorflowMusiCNN(tf, modelPath);
    await musiCNN.initialize();

    const fakeFeatures = {
      melSpectrum: getZeroMatrix(187, 96),
      frameSize: 187,
      melBandsSize: 96,
      patchSize: 187
    };

    const fakeStart = Date.now();

    musiCNN.predict(fakeFeatures, false).then((predictions) => {
      const summarizedPredictions = twoValuesAverage(predictions);
      console.log(summarizedPredictions)
      // const results = summarizedPredictions.filter((_, i) => modelTagOrder[modelName][i])[0];
      // console.log(results)


      console.info(`Model: Warm up inference took: ${Date.now() - fakeStart}`);
      let modelReady = true;
      if (modelReady) console.log(`Model loaded and ready.`);
    });

    console.log(musiCNN)

    // Log the extracted features
    console.log('Extracted Features:');
    console.log({
      danceability,
      duration,
      energy,
      key,
      mode,
      loudness,
      tempo,
    });
  } catch (err) {
    console.error('Error processing audio file:', err);
  }
};

function getZeroMatrix(x, y) {
  let matrix = new Array(x);
  for (let f = 0; f < x; f++) {
      matrix[f] = new Array(y).fill(0);
  }
  return matrix;
}

function twoValuesAverage (arrayOfArrays) {
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

// Run the main function
main();

