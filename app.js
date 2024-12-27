import express from 'express';
import * as tf from '@tensorflow/tfjs-node';
import pkg from 'essentia.js';

const { Essentia, EssentiaWASM, EssentiaModel } = pkg;


// Initialize Essentia
const essentia = new Essentia(EssentiaWASM);

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

// Simple route to test TensorFlow operations
app.post('/predict', async (req, res) => {
    try {
        // Example: Create a simple tensor and perform an operation
        const tensor = tf.tensor2d([[1, 2], [3, 4]]);
        const result = tensor.add(tf.scalar(1));
        
        res.json({
            input: tensor.arraySync(),
            output: result.arraySync()
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/health', (req, res) => {
    res.json({ status: 'OK', tfVersion: tf.version });
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});