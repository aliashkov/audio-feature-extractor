import { workerData, parentPort } from 'worker_threads';
import { Essentia, EssentiaWASM, EssentiaModel } from 'essentia.js';
import decode from 'audio-decode';
import fetch from 'node-fetch';

const essentia = new Essentia(EssentiaWASM);
const extractor = new EssentiaModel.EssentiaTFInputExtractor(EssentiaWASM, 'musicnn', false);

// Define maximum buffer size (10MB = 10 * 1024 * 1024 bytes)
const MAX_BUFFER_SIZE = 10 * 1024 * 1024;

async function computeFeatures(offlineUrl) {
  const response = await fetch(offlineUrl, {
    headers: {
      'Range': 'bytes=0-10485759' // Request the first 10MB (10 * 1024 * 1024 bytes)
    }
  });
  
  // Get content length from headers
  const contentLength = parseInt(response.headers.get('content-length'));
  
  if (contentLength && contentLength > MAX_BUFFER_SIZE) {
    throw new Error(`File size exceeds maximum limit of 10MB (actual size: ${(contentLength / (1024 * 1024)).toFixed(2)}MB)`);
  }
  
  const buffer = await response.arrayBuffer();
  
  // Double-check actual buffer size
  if (buffer.byteLength > MAX_BUFFER_SIZE) {
    throw new Error(`File size exceeds maximum limit of 10MB (actual size: ${(buffer.byteLength / (1024 * 1024)).toFixed(2)}MB)`);
  }

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
    const featuresData = await computeFeatures(workerData.offlineUrl);

    // Send features back to main thread
    parentPort.postMessage({
      type: 'analyze',
      featuresData: featuresData
    });
  } catch (error) {
    console.error('Error in worker:', error);
    parentPort.postMessage({
      type: 'error',
      error: error.message || 'Error processing audio'
    });
  }
}

run();