import { initModels } from './main.js';


initModels().then(() => {
  console.log('Models initialized successfully!');
  process.exit(0);
}).catch((err) => {
  console.error('Error initializing models:', err);
  process.exit(1);
});