import { Essentia, EssentiaWASM, EssentiaModel } from 'essentia.js';
import * as tf from '@tensorflow/tfjs-node';

const modelPaths = {
  dancability: 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/danceability-musicnn-msd-2/model.json',
  happy: 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/mood_happy-musicnn-msd-2/model.json',
  sad: 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/mood_sad-musicnn-msd-2/model.json',
  relaxed: 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/mood_relaxed-musicnn-msd-2/model.json',
  aggressive: 'https://raw.githubusercontent.com/MTG/essentia.js/refs/heads/master/examples/demos/mood-classifiers/models/mood_aggressive-musicnn-msd-2/model.json',
};

// Initialize Essentia
// const essentia = new Essentia(EssentiaWASM);

// Function to initialize models
export async function initModels() {
  const models = {};
  for (const [name, path] of Object.entries(modelPaths)) {
    const { TensorflowMusiCNN } = await import('essentia.js/dist/essentia.js-model.umd.js');
    models[name] = new TensorflowMusiCNN(tf, path);
    await models[name].initialize();
    console.info(`Model ${name} has been loaded!`);
  }
  return models; // Return models object
}