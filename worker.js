import { workerData, parentPort } from 'worker_threads';
import { Essentia, EssentiaWASM, EssentiaModel } from 'essentia.js';
import decode from 'audio-decode';
import fetch from 'node-fetch';

const essentia = new Essentia(EssentiaWASM);
const extractor = new EssentiaModel.EssentiaTFInputExtractor(EssentiaWASM, 'musicnn', false);

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