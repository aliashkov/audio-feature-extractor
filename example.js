import fs from 'fs';
import decode from 'audio-decode';
import { Essentia, EssentiaWASM, EssentiaModel } from 'essentia.js';

// Initialize Essentia
const essentia = new Essentia(EssentiaWASM);

// Model paths
const modelPaths = {
  mood_dancability: 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/danceability-musicnn-msd-2/model.json',
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

    // Array of audio URLs
    const audioUrls = [
      'https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002ad4c0-96d8-4eb1-827c-323a8192c045/002ad4c0-96d8-4eb1-827c-323a8192c045.mp3',
      // 'https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002b667c-a888-42d5-8ef4-487d8f0495f5/002b667c-a888-42d5-8ef4-487d8f0495f5.mp3',
      //'https://link.storjshare.io/raw/jvunpvyh2hogqgydf4dspxkupzma/tracks/002a94f2-37e6-4d63-bbd2-6da61a511262/002a94f2-37e6-4d63-bbd2-6da61a511262.mp3'
    ];

    for (const audioUrl of audioUrls) {
      try {
        const response = await fetch(audioUrl);
        const buffer = await response.arrayBuffer();

        const audio = await decode(buffer);
        const data = essentia.arrayToVector(audio._channelData[0]);

        // Extract audio features
        const duration = essentia.Duration(data).duration;
        const energy = essentia.Energy(data).energy;
        const keyExtractor = essentia.KeyExtractor(data);
        const KEYS = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
        const key = KEYS.indexOf(keyExtractor.key);
        const mode = keyExtractor.scale === 'major' ? 1 : 0;
        const loudness = essentia.DynamicComplexity(data).loudness;
        const tempo = essentia.PercivalBpmEstimator(data).bpm;

        console.log(`Processing ${audioUrl} - Duration: ${duration}, Energy: ${energy}, Key: ${key}, Mode: ${mode}, Loudness: ${loudness}, Tempo: ${tempo}`);

        // Prepare features for prediction
        const features = extractor.computeFrameWise(audio._channelData[0], 256); // Adjust frame size here if needed

        // Predict using all models
        for (const modelName of Object.keys(models)) {
          const selectedModel = models[modelName]; // Use the current model
          const predictions = await selectedModel.predict(features, true);

          // Adjust predictions for 'relaxed' and 'sad'
          let summarizedPredictions = twoValuesAverage(predictions);
          
          if (modelName === 'mood_relaxed' || modelName === 'mood_sad') {
            summarizedPredictions = summarizedPredictions.map(value => 1 - value); // Reverse the predictions
          }

          console.log(`Predictions for ${modelName}:`, summarizedPredictions);
        }

      } catch (err) {
        console.error(`Error processing audio file ${audioUrl}:`, err);
      }
    }

  } catch (err) {
    console.error('Error initializing TensorFlow.js or models:', err);
  }
};

// Run the main function
main();
