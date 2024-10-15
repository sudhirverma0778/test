const readline = require('readline');
const tf = require('@tensorflow/tfjs');

// Set up readline interface
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

// Function to get user inputs for the matrix
function getMatrixInput(callback) {
    const matrix = [];
    let count = 0;

    const askForInput = () => {
        if (count < 36) { // 9 rows x 4 columns = 36 inputs
            rl.question(`Enter value for element ${Math.floor(count / 4) + 1},${(count % 4) + 1}: `, (input) => {
                matrix.push(parseFloat(input));
                count++;
                askForInput();
            });
        } else {
            rl.close();
            callback(matrix);
        }
    };
    askForInput();
}

// Function to create and train the model
async function trainModel(data) {
    const xs = tf.tensor2d(data, [9, 4]);

    // Placeholder for outputs; for demonstration, we can just use random values
    const ys = tf.randomUniform([9, 20]);

    const model = tf.sequential();
    model.add(tf.layers.dense({units: 10, activation: 'relu', inputShape: [4]}));
    model.add(tf.layers.dense({units: 20}));

    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

    await model.fit(xs, ys, {epochs: 100});
    return model;
}

// Function to predict outcomes
async function predictOutcomes(model) {
    // Use the last row of the input matrix for prediction
    const input = tf.tensor2d([[...Array(4)].map(() => Math.random())]); // Random input for prediction
    const prediction = model.predict(input);
	console.log("prediction", prediction);
    Math.floor(prediction.print());
}

// Main function
async function main() {
    getMatrixInput(async (matrix) => {
		console.log("matrix", matrix);
        const model = await trainModel(matrix);
        await predictOutcomes(model);
    });
}

main();
