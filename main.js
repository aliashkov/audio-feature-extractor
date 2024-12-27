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

async function predict(audioUrl) {
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
    const predictions = {};
    for (const modelName of Object.keys(models)) {
      const selectedModel = models[modelName]; // Use the current model
      const predictionsArray = await selectedModel.predict(features, true);

      // Adjust predictions for 'relaxed' and 'sad'
      let summarizedPredictions = twoValuesAverage(predictionsArray);

      if (modelName === 'mood_relaxed' || modelName === 'mood_sad') {
        summarizedPredictions = summarizedPredictions.map(value => 1 - value); // Reverse the predictions
      }

      predictions[modelName] = summarizedPredictions;
    }

    return predictions;
  } catch (err) {
    console.error(`Error processing audio file ${audioUrl}:`, err);
    throw err;
  }
}

const models = initModels(tf);

export { predict, initModels };
