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
  
    for (const modelName of Object.keys(models)) {
      const selectedModel = models[modelName];
  
      const predictionsArray = await selectedModel.predict(featuresData.features, true);
  
      let summarizedPredictions = twoValuesAverage(predictionsArray);
      if (modelName === 'mood_relaxed' || modelName === 'mood_sad') {
        summarizedPredictions = summarizedPredictions.map(value => 1 - value);
      }
  
      const prediction = summarizedPredictions[0];
      predictions[modelName] = prediction;
    }
  
    const formattedPredictions = {
      ...predictions,
      energy: featuresData.energy,
      loudness: featuresData.loudness,
      tempo: featuresData.tempo,
    };
  
    return formattedPredictions;
  }