import fs from 'fs';
import decode from 'audio-decode';
import { Essentia, EssentiaWASM, EssentiaModel } from 'essentia.js';

const KEEP_PERCENTAGE = 0.15; // keep only 15% of audio file

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

const extractor = new EssentiaModel.EssentiaTFInputExtractor(EssentiaWASM, 'musicnn', false);

async function loadTensorflowWASM() {
  const tf = await import('@tensorflow/tfjs-node');
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

    const buffer = fs.readFileSync('audio/3.mp3');

    console.log(buffer)

    const audio = await decode(buffer);

    // console.log(audio)



    const data = essentia.arrayToVector(audio._channelData[0]);

    console.log(audio._channelData[0])

    //let audioData2 = shortenAudio(audio, KEEP_PERCENTAGE, true); // <-- TRIMMED start/end

    // console.log(audioData2)


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

    // let audioData = shortenAudio(prepocessedAudio, KEEP_PERCENTAGE, true); // <-- TRIMMED start/end


    const features = extractor.computeFrameWise(audio._channelData[0], 256); // Adjust frame size here if needed

    console.log(features)


    //console.log(fakeFeatures)

    const fakeStart = Date.now();
    const predictions = await model.predict(features, true);
    const summarizedPredictions = twoValuesAverage(predictions);
    // console.log('Predictions:', summarizedPredictions);
    console.log(predictions)
    console.info(`Model: Warm up inference took: ${Date.now() - fakeStart}`);

  } catch (err) {
    console.error('Error processing audio file:', err);
  }
};

function shortenAudio (audioIn, keepRatio=0.5, trim=false) {
  /* 
      keepRatio applied after discarding start and end (if trim == true)
  */
  if (keepRatio < 0.15) {
      keepRatio = 0.15 // must keep at least 15% of the file
  } else if (keepRatio > 0.66) {
      keepRatio = 0.66 // will keep at most 2/3 of the file
  }

  if (trim) {
      const discardSamples = Math.floor(0.1 * audioIn.length); // discard 10% on beginning and end
      audioIn = audioIn.subarray(discardSamples, audioIn.length - discardSamples); // create new view of buffer without beginning and end
  }

  const ratioSampleLength = Math.ceil(audioIn.length * keepRatio);
  const patchSampleLength = 187 * 256; // cut into patchSize chunks so there's no weird jumps in audio
  const numPatchesToKeep = Math.ceil(ratioSampleLength / patchSampleLength);

  // space patchesToKeep evenly
  const skipSize = Math.floor( (audioIn.length - ratioSampleLength) / (numPatchesToKeep - 1) );

  let audioOut = [];
  let startIndex = 0;
  for (let i = 0; i < numPatchesToKeep; i++) {
      let endIndex = startIndex + patchSampleLength;
      let chunk = audioIn.slice(startIndex, endIndex);
      audioOut.push(...chunk);
      startIndex = endIndex + skipSize; // discard even space
  }

  return Float32Array.from(audioOut);
}

// Run the main function
main();