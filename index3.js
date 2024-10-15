const tf = require('@tensorflow/tfjs');
const readlineSync = require('readline-sync');

// Function to collect user input for a 9x4 matrix
function collectInputs(numSamples = 5, rows = 9, cols = 4) {
    const data = [];
    console.log(`Please enter ${numSamples} iterations of ${rows} rows and ${cols} columns:`);
    
    for (let i = 0; i < numSamples; i++) {
        const matrix = [];
        for (let j = 0; j < rows; j++) {
            const row = readlineSync.question(`Enter values for row ${j + 1} (space-separated): `)
                .split(' ')
                .map(Number);
            if (row.length !== cols) {
                console.log(`Please enter exactly ${cols} numbers.`);
                j--; // repeat this row
            } else {
                matrix.push(row);
            }
        }
        data.push(matrix.flat());

    }
    
    return data;
}

// Step 1: Collect user inputs
const data = collectInputs();

// Step 2: Prepare input features (X) and output targets (y)
const X = data.slice(0, 5).map(row => row);
const y = data.slice(5, 25).map(row => row); // Assuming 20 future predictions based on 5 samples

// Step 3: Convert to TensorFlow tensors
const inputTensor = tf.tensor2d(X);
const outputTensor = tf.tensor2d(y);

// Step 4: Create and train the model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 20, inputShape: [36], activation: 'relu' })); // 9x4 = 36
model.add(tf.layers.dense({ units: 36, activation: 'relu' }));
model.add(tf.layers.dense({ units: 36, activation: 'linear' }));

model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

(async () => {
    await model.fit(inputTensor, outputTensor, {
        epochs: 500,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
            }
        }
    });

    // Step 5: Predict the next 20 iterations
    const predictions = model.predict(inputTensor).arraySync();

    // Step 6: Output the predictions
    console.log("\nPredicted Outputs for the Next 20 Iterations:");
    predictions.forEach((pred, i) => {
        console.log(`Prediction for iteration ${i + 1}: ${pred}`);
    });
})();
