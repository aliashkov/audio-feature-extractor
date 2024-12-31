import * as tf from '@tensorflow/tfjs-node';


export function twoValuesAverage(arrayOfArrays) {
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

export async function predict(featuresData, models) {
  const predictions = {};
  
  try {
    for (const modelName of Object.keys(models)) {
      const selectedModel = models[modelName];
      
      // Perform prediction
      const predictionsArray = await selectedModel.predict(featuresData.features, true);
      
      // Handle different types of prediction results
      let predictionsData;
      if (predictionsArray instanceof tf.Tensor) {
        // If it's a tensor, use dataSync() or await data()
        predictionsData = Array.from(predictionsArray.dataSync());
        predictionsArray.dispose(); // Clean up tensor
      } else if (Array.isArray(predictionsArray)) {
        // If it's already an array
        predictionsData = predictionsArray;
      } else {
        // If it's a single value
        predictionsData = [predictionsArray];
      }
      
      let summarizedPredictions = twoValuesAverage(predictionsData);
      if (modelName === 'mood_relaxed' || modelName === 'mood_sad') {
        summarizedPredictions = summarizedPredictions.map(value => 1 - value);
      }
      
      predictions[modelName] = summarizedPredictions[0];
    }
    
    return {
      ...predictions,
      energy: featuresData.energy,
      loudness: featuresData.loudness,
      tempo: featuresData.tempo,
    };
  } catch (error) {
    throw error;
  }
}