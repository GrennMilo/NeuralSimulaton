// Neural Network Simulation JavaScript

// Global variables
let isPlaying = false;
let epochCounter = 0;
let animationSpeed = 30; // ms between epochs during animation
let timer;
let network = []; // Store neural network structure
let weights = []; // Store weights between layers
let neurons = []; // Store neuron values
let biases = []; // Store bias values
let dataPoints = []; // Store generated data points
let testData = []; // Store test data points
let trainingData = []; // Store training data points

// Neural network parameters
const params = {
    learningRate: 0.03,
    activation: 'relu',
    regularization: 'none',
    regularizationRate: 0,
    problemType: 'classification',
    dataset: 'circle',
    trainRatio: 50,
    noise: 0,
    batchSize: 10,
    hiddenLayers: 2,
    neuronsPerLayer: 3,
    features: ['x1', 'x2'] // Default active features
};

// Color scales for visualization
const orangeBlueScale = d3.scaleLinear()
    .domain([-1, 0, 1])
    .range(['#ff9800', '#ffffff', '#2196f3']);

// DOM elements
document.addEventListener('DOMContentLoaded', () => {
    // Initialize UI controls
    initializeControls();
    
    // Generate initial dataset
    generateData();
    
    // Initialize neural network
    initializeNetwork();
    
    // Render visualizations
    renderDatasetPreviews();
    renderFeaturePreviews();
    renderNetwork();
    renderOutputVisualization();
});

// Initialize UI controls and event listeners
function initializeControls() {
    // Playback controls
    document.getElementById('play').addEventListener('click', togglePlayback);
    document.getElementById('step').addEventListener('click', stepForward);
    document.getElementById('reset').addEventListener('click', resetNetwork);
    
    // Parameter controls
    document.getElementById('learning-rate').addEventListener('change', e => {
        params.learningRate = parseFloat(e.target.value);
    });
    
    document.getElementById('activation').addEventListener('change', e => {
        params.activation = e.target.value;
    });
    
    document.getElementById('regularization').addEventListener('change', e => {
        params.regularization = e.target.value;
    });
    
    document.getElementById('regularization-rate').addEventListener('change', e => {
        params.regularizationRate = parseFloat(e.target.value);
    });
    
    document.getElementById('problem-type').addEventListener('change', e => {
        params.problemType = e.target.value;
        renderOutputVisualization();
    });
    
    // Dataset controls
    document.querySelectorAll('.dataset').forEach(el => {
        el.addEventListener('click', () => {
            document.querySelector('.dataset.selected').classList.remove('selected');
            el.classList.add('selected');
            params.dataset = el.getAttribute('data-set');
            generateData();
            resetNetwork();
            renderOutputVisualization();
        });
    });
    
    // Data parameter controls
    document.getElementById('train-ratio').addEventListener('input', e => {
        params.trainRatio = parseInt(e.target.value);
        document.getElementById('train-ratio-value').textContent = `${params.trainRatio}%`;
        splitTrainTestData();
        renderOutputVisualization();
    });
    
    document.getElementById('noise').addEventListener('input', e => {
        params.noise = parseInt(e.target.value);
        document.getElementById('noise-value').textContent = params.noise;
        generateData();
        resetNetwork();
        renderOutputVisualization();
    });
    
    document.getElementById('batch-size').addEventListener('input', e => {
        params.batchSize = parseInt(e.target.value);
        document.getElementById('batch-size-value').textContent = params.batchSize;
    });
    
    // Regenerate button
    document.getElementById('regenerate').addEventListener('click', () => {
        generateData();
        resetNetwork();
        renderOutputVisualization();
    });
    
    // Layer controls
    document.getElementById('add-layer').addEventListener('click', () => {
        if (params.hiddenLayers < 5) {
            params.hiddenLayers++;
            document.getElementById('layer-count').textContent = params.hiddenLayers;
            initializeNetwork();
            renderNetwork();
            renderOutputVisualization();
        }
    });
    
    document.getElementById('remove-layer').addEventListener('click', () => {
        if (params.hiddenLayers > 1) {
            params.hiddenLayers--;
            document.getElementById('layer-count').textContent = params.hiddenLayers;
            initializeNetwork();
            renderNetwork();
            renderOutputVisualization();
        }
    });
    
    // Output options
    document.getElementById('show-test-data').addEventListener('change', () => {
        renderOutputVisualization();
    });
    
    document.getElementById('discretize-output').addEventListener('change', () => {
        renderOutputVisualization();
    });
}

// Data generation functions
function generateData() {
    dataPoints = [];
    const numPoints = 200;
    
    switch (params.dataset) {
        case 'circle':
            generateCircleData(numPoints);
            break;
        case 'spiral':
            generateSpiralData(numPoints);
            break;
        case 'xor':
            generateXORData(numPoints);
            break;
        case 'gaussian':
            generateGaussianData(numPoints);
            break;
    }
    
    // Add noise if specified
    if (params.noise > 0) {
        addNoise();
    }
    
    // Split into training and test sets
    splitTrainTestData();
}

function generateCircleData(numPoints) {
    for (let i = 0; i < numPoints; i++) {
        // Generate points in a square from -5 to 5
        const x = (Math.random() * 10) - 5;
        const y = (Math.random() * 10) - 5;
        const distFromCenter = Math.sqrt(x*x + y*y);
        
        // Points inside circle (radius 3) are class 1, outside are class 0
        const label = distFromCenter < 3 ? 1 : 0;
        
        dataPoints.push({
            x: x,
            y: y,
            label: label
        });
    }
}

function generateSpiralData(numPoints) {
    const pointsPerClass = Math.floor(numPoints / 2);
    
    // Generate two spirals
    for (let i = 0; i < pointsPerClass; i++) {
        // First spiral (class 0)
        const r1 = (i / pointsPerClass) * 5;
        const t1 = 1.75 * i / pointsPerClass * 2 * Math.PI + Math.PI;
        const x1 = r1 * Math.sin(t1);
        const y1 = r1 * Math.cos(t1);
        
        // Second spiral (class 1)
        const r2 = (i / pointsPerClass) * 5;
        const t2 = 1.75 * i / pointsPerClass * 2 * Math.PI;
        const x2 = r2 * Math.sin(t2);
        const y2 = r2 * Math.cos(t2);
        
        dataPoints.push({
            x: x1,
            y: y1,
            label: 0
        });
        
        dataPoints.push({
            x: x2,
            y: y2,
            label: 1
        });
    }
}

function generateXORData(numPoints) {
    for (let i = 0; i < numPoints; i++) {
        // Generate points in a square from -5 to 5
        const x = (Math.random() * 10) - 5;
        const y = (Math.random() * 10) - 5;
        
        // XOR pattern: both positive or both negative is class 0, otherwise class 1
        const label = (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : 1;
        
        dataPoints.push({
            x: x,
            y: y,
            label: label
        });
    }
}

function generateGaussianData(numPoints) {
    const pointsPerCluster = Math.floor(numPoints / 2);
    
    // Cluster 1 (class 0)
    for (let i = 0; i < pointsPerCluster; i++) {
        const x = randomGaussian(-2.5, 1);
        const y = randomGaussian(-2.5, 1);
        
        dataPoints.push({
            x: x,
            y: y,
            label: 0
        });
    }
    
    // Cluster 2 (class 1)
    for (let i = 0; i < pointsPerCluster; i++) {
        const x = randomGaussian(2.5, 1);
        const y = randomGaussian(2.5, 1);
        
        dataPoints.push({
            x: x,
            y: y,
            label: 1
        });
    }
}

// Helper for Gaussian distribution
function randomGaussian(mean, stdDev) {
    const u1 = 1 - Math.random();
    const u2 = 1 - Math.random();
    const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
    return mean + stdDev * randStdNormal;
}

function addNoise() {
    const noiseAmount = params.noise / 100;
    const numPointsToFlip = Math.floor(dataPoints.length * noiseAmount);
    
    // Randomly flip labels for noise percentage of points
    for (let i = 0; i < numPointsToFlip; i++) {
        const idx = Math.floor(Math.random() * dataPoints.length);
        dataPoints[idx].label = 1 - dataPoints[idx].label; // Flip the label (0 -> 1, 1 -> 0)
    }
}

function splitTrainTestData() {
    const trainRatio = params.trainRatio / 100;
    
    // Shuffle the data points
    const shuffled = [...dataPoints].sort(() => Math.random() - 0.5);
    
    // Split into training and test sets
    const splitIndex = Math.floor(shuffled.length * trainRatio);
    trainingData = shuffled.slice(0, splitIndex);
    testData = shuffled.slice(splitIndex);
}

// Neural network initialization and training functions
function initializeNetwork() {
    // Reset network structure
    network = [];
    weights = [];
    biases = [];
    neurons = [];
    epochCounter = 0;
    document.getElementById('epoch-value').textContent = epochCounter;
    
    // Input layer based on active features
    const inputSize = params.features.length;
    network.push(inputSize);
    
    // Hidden layers
    for (let i = 0; i < params.hiddenLayers; i++) {
        network.push(params.neuronsPerLayer);
    }
    
    // Output layer (1 neuron for binary classification/regression)
    network.push(1);
    
    // Initialize weights and biases with random values
    for (let i = 0; i < network.length - 1; i++) {
        const currentLayerSize = network[i];
        const nextLayerSize = network[i + 1];
        
        // Initialize weights between current layer and next layer
        const layerWeights = [];
        for (let j = 0; j < currentLayerSize; j++) {
            const neuronWeights = [];
            for (let k = 0; k < nextLayerSize; k++) {
                // Xavier/Glorot initialization
                const limit = Math.sqrt(6 / (currentLayerSize + nextLayerSize));
                neuronWeights.push((Math.random() * 2 * limit) - limit);
            }
            layerWeights.push(neuronWeights);
        }
        weights.push(layerWeights);
        
        // Initialize biases for next layer
        const layerBiases = [];
        for (let j = 0; j < nextLayerSize; j++) {
            layerBiases.push(0); // Initialize biases to zero
        }
        biases.push(layerBiases);
    }
    
    // Initialize neurons for all layers
    for (let i = 0; i < network.length; i++) {
        const layerNeurons = [];
        for (let j = 0; j < network[i]; j++) {
            layerNeurons.push(0);
        }
        neurons.push(layerNeurons);
    }
}

// Forward propagation
function forwardPropagate(inputs) {
    // Set input layer values
    for (let i = 0; i < inputs.length; i++) {
        neurons[0][i] = inputs[i];
    }
    
    // Forward propagate through the network
    for (let i = 1; i < network.length; i++) {
        const prevLayer = i - 1;
        
        for (let j = 0; j < network[i]; j++) {
            let sum = biases[prevLayer][j];
            
            // Calculate weighted sum of inputs
            for (let k = 0; k < network[prevLayer]; k++) {
                sum += neurons[prevLayer][k] * weights[prevLayer][k][j];
            }
            
            // Apply activation function
            neurons[i][j] = activate(sum);
        }
    }
    
    // Return the output
    return neurons[neurons.length - 1][0];
}

// Activation functions
function activate(x) {
    switch (params.activation) {
        case 'relu':
            return relu(x);
        case 'sigmoid':
            return sigmoid(x);
        case 'tanh':
            return tanh(x);
        default:
            return sigmoid(x);
    }
}

function relu(x) {
    return Math.max(0, x);
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function tanh(x) {
    return Math.tanh(x);
}

// Derivatives of activation functions for backpropagation
function activateDerivative(x) {
    switch (params.activation) {
        case 'relu':
            return reluDerivative(x);
        case 'sigmoid':
            return sigmoidDerivative(x);
        case 'tanh':
            return tanhDerivative(x);
        default:
            return sigmoidDerivative(x);
    }
}

function reluDerivative(x) {
    return x > 0 ? 1 : 0;
}

function sigmoidDerivative(x) {
    const s = sigmoid(x);
    return s * (1 - s);
}

function tanhDerivative(x) {
    const t = tanh(x);
    return 1 - t * t;
}

// Backpropagation
function backPropagate(inputs, target) {
    // Forward pass to calculate all neuron values
    forwardPropagate(inputs);
    
    // Calculate output error
    const outputLayer = neurons.length - 1;
    const outputValue = neurons[outputLayer][0];
    const outputError = outputValue - target;
    
    // Calculate gradients and deltas for all layers, starting from the output
    const deltas = [];
    for (let i = 0; i < network.length; i++) {
        const layerDeltas = [];
        for (let j = 0; j < network[i]; j++) {
            layerDeltas.push(0);
        }
        deltas.push(layerDeltas);
    }
    
    // Output layer gradient
    deltas[outputLayer][0] = outputError;
    
    // Hidden layers gradients (backpropagate the error)
    for (let i = outputLayer - 1; i > 0; i--) {
        for (let j = 0; j < network[i]; j++) {
            let error = 0;
            for (let k = 0; k < network[i + 1]; k++) {
                error += weights[i][j][k] * deltas[i + 1][k];
            }
            deltas[i][j] = error * activateDerivative(neurons[i][j]);
        }
    }
    
    // Update weights and biases
    for (let i = 0; i < outputLayer; i++) {
        for (let j = 0; j < network[i]; j++) {
            for (let k = 0; k < network[i + 1]; k++) {
                // Calculate weight regularization term
                let regularizationTerm = 0;
                if (params.regularization === 'l1') {
                    regularizationTerm = params.regularizationRate * Math.sign(weights[i][j][k]);
                } else if (params.regularization === 'l2') {
                    regularizationTerm = params.regularizationRate * weights[i][j][k];
                }
                
                // Update weight
                weights[i][j][k] -= params.learningRate * (neurons[i][j] * deltas[i + 1][k] + regularizationTerm);
            }
        }
        
        // Update biases
        for (let j = 0; j < network[i + 1]; j++) {
            biases[i][j] -= params.learningRate * deltas[i + 1][j];
        }
    }
    
    return outputError * outputError; // Return squared error
}

// Train network for one epoch
function trainEpoch() {
    let totalError = 0;
    
    // Use mini-batch gradient descent
    const batchSize = Math.min(params.batchSize, trainingData.length);
    
    // Shuffle training data and take a batch
    const batch = [...trainingData].sort(() => Math.random() - 0.5).slice(0, batchSize);
    
    // Train on each data point in the batch
    for (const point of batch) {
        const inputs = getFeatureValues(point);
        const target = point.label;
        
        const error = backPropagate(inputs, target);
        totalError += error;
    }
    
    // Increment epoch counter
    epochCounter++;
    document.getElementById('epoch-value').textContent = epochCounter;
    
    // Calculate average error
    const avgError = totalError / batchSize;
    
    // Calculate test error
    const testError = calculateTestError();
    
    // Update display
    document.getElementById('training-loss').textContent = avgError.toFixed(3);
    document.getElementById('test-loss').textContent = testError.toFixed(3);
    
    // Update visualizations
    renderNetwork();
    renderOutputVisualization();
    
    return avgError;
}

// Calculate error on test data
function calculateTestError() {
    let totalError = 0;
    
    for (const point of testData) {
        const inputs = getFeatureValues(point);
        const target = point.label;
        
        const prediction = forwardPropagate(inputs);
        const error = prediction - target;
        totalError += error * error;
    }
    
    return totalError / testData.length;
}

// Get feature values based on active features
function getFeatureValues(point) {
    const values = [];
    
    for (const feature of params.features) {
        switch (feature) {
            case 'x1':
                values.push(point.x);
                break;
            case 'x2':
                values.push(point.y);
                break;
            case 'x1^2':
                values.push(point.x * point.x);
                break;
            case 'x2^2':
                values.push(point.y * point.y);
                break;
            case 'x1x2':
                values.push(point.x * point.y);
                break;
            case 'sin(x1)':
                values.push(Math.sin(point.x));
                break;
            case 'sin(x2)':
                values.push(Math.sin(point.y));
                break;
        }
    }
    
    return values;
}

// Playback controls
function togglePlayback() {
    isPlaying = !isPlaying;
    
    if (isPlaying) {
        document.getElementById('play').innerHTML = '<svg viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>';
        timer = setInterval(() => {
            trainEpoch();
        }, animationSpeed);
    } else {
        document.getElementById('play').innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
        clearInterval(timer);
    }
}

function stepForward() {
    if (!isPlaying) {
        trainEpoch();
    }
}

function resetNetwork() {
    if (isPlaying) {
        togglePlayback();
    }
    
    initializeNetwork();
    renderNetwork();
    renderOutputVisualization();
}

// Rendering functions
function renderDatasetPreviews() {
    renderDatasetPreview('circle-preview', generateCirclePreviewData);
    renderDatasetPreview('spiral-preview', generateSpiralPreviewData);
    renderDatasetPreview('xor-preview', generateXORPreviewData);
    renderDatasetPreview('gaussian-preview', generateGaussianPreviewData);
}

function renderDatasetPreview(canvasId, dataGeneratorFunc) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);
    
    // Generate preview data
    const previewData = dataGeneratorFunc(40);
    
    // Render data points
    for (const point of previewData) {
        ctx.fillStyle = point.label === 1 ? '#2196f3' : '#ff9800';
        
        // Map coordinates to canvas
        const x = (point.x + 5) / 10 * width;
        const y = (point.y + 5) / 10 * height;
        
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 2 * Math.PI);
        ctx.fill();
    }
}

function generateCirclePreviewData(numPoints) {
    const data = [];
    
    for (let i = 0; i < numPoints; i++) {
        const x = (Math.random() * 10) - 5;
        const y = (Math.random() * 10) - 5;
        const distFromCenter = Math.sqrt(x*x + y*y);
        
        data.push({
            x: x,
            y: y,
            label: distFromCenter < 3 ? 1 : 0
        });
    }
    
    return data;
}

function generateSpiralPreviewData(numPoints) {
    const data = [];
    const pointsPerClass = Math.floor(numPoints / 2);
    
    for (let i = 0; i < pointsPerClass; i++) {
        const r1 = (i / pointsPerClass) * 5;
        const t1 = 1.75 * i / pointsPerClass * 2 * Math.PI + Math.PI;
        const x1 = r1 * Math.sin(t1);
        const y1 = r1 * Math.cos(t1);
        
        const r2 = (i / pointsPerClass) * 5;
        const t2 = 1.75 * i / pointsPerClass * 2 * Math.PI;
        const x2 = r2 * Math.sin(t2);
        const y2 = r2 * Math.cos(t2);
        
        data.push({ x: x1, y: y1, label: 0 });
        data.push({ x: x2, y: y2, label: 1 });
    }
    
    return data;
}

function generateXORPreviewData(numPoints) {
    const data = [];
    
    for (let i = 0; i < numPoints; i++) {
        const x = (Math.random() * 10) - 5;
        const y = (Math.random() * 10) - 5;
        
        data.push({
            x: x,
            y: y,
            label: (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : 1
        });
    }
    
    return data;
}

function generateGaussianPreviewData(numPoints) {
    const data = [];
    const pointsPerCluster = Math.floor(numPoints / 2);
    
    for (let i = 0; i < pointsPerCluster; i++) {
        data.push({
            x: randomGaussian(-2.5, 1),
            y: randomGaussian(-2.5, 1),
            label: 0
        });
        
        data.push({
            x: randomGaussian(2.5, 1),
            y: randomGaussian(2.5, 1),
            label: 1
        });
    }
    
    return data;
}

function renderFeaturePreviews() {
    const previewData = [...dataPoints].slice(0, 50);
    
    renderFeaturePreview('x1-preview', previewData, point => point.x, 'x1');
    renderFeaturePreview('x2-preview', previewData, point => point.y, 'x2');
    renderFeaturePreview('x1-squared-preview', previewData, point => point.x * point.x, 'x1^2');
    renderFeaturePreview('x2-squared-preview', previewData, point => point.y * point.y, 'x2^2');
    renderFeaturePreview('x1x2-preview', previewData, point => point.x * point.y, 'x1x2');
    renderFeaturePreview('sin-x1-preview', previewData, point => Math.sin(point.x), 'sin(x1)');
    renderFeaturePreview('sin-x2-preview', previewData, point => Math.sin(point.y), 'sin(x2)');
}

function renderFeaturePreview(canvasId, data, valueFn, featureName) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);
    
    // Calculate min and max values
    let minVal = Infinity;
    let maxVal = -Infinity;
    
    for (const point of data) {
        const val = valueFn(point);
        minVal = Math.min(minVal, val);
        maxVal = Math.max(maxVal, val);
    }
    
    // Add padding to range
    const padding = (maxVal - minVal) * 0.1;
    minVal -= padding;
    maxVal += padding;
    
    // Normalize to [-1, 1] range for color mapping
    const normalizeValue = val => {
        if (minVal === maxVal) return 0;
        return 2 * ((val - minVal) / (maxVal - minVal)) - 1;
    };
    
    // Draw colored squares for each data point
    const squareSize = 6;
    const cols = Math.floor(Math.sqrt(data.length));
    const rows = Math.ceil(data.length / cols);
    
    for (let i = 0; i < data.length; i++) {
        const row = Math.floor(i / cols);
        const col = i % cols;
        
        const x = col * squareSize;
        const y = row * squareSize;
        
        const value = valueFn(data[i]);
        const normalizedValue = normalizeValue(value);
        
        ctx.fillStyle = orangeBlueScale(normalizedValue);
        ctx.fillRect(x, y, squareSize, squareSize);
    }
    
    // Highlight border if feature is active
    if (params.features.includes(featureName)) {
        ctx.strokeStyle = '#4a86e8';
        ctx.lineWidth = 2;
        ctx.strokeRect(0, 0, width, height);
    }
    
    // Add click event to toggle feature
    canvas.parentNode.addEventListener('click', () => {
        const index = params.features.indexOf(featureName);
        
        if (index === -1) {
            params.features.push(featureName);
        } else {
            params.features.splice(index, 1);
        }
        
        renderFeaturePreview(canvasId, data, valueFn, featureName);
        initializeNetwork();
        renderNetwork();
        renderOutputVisualization();
    });
}

function renderNetwork() {
    const container = document.getElementById('network-architecture');
    
    // Clear container
    container.innerHTML = '';
    
    // Create SVG element
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.style.overflow = 'visible';
    container.appendChild(svg);
    
    // Get container dimensions
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Calculate layer positions
    const layerSpacing = width / (network.length + 1);
    const maxNeurons = Math.max(...network);
    const neuronSpacing = height / (maxNeurons + 1);
    const neuronRadius = 20;
    
    // Draw connections between neurons first (so they appear behind)
    for (let i = 0; i < network.length - 1; i++) {
        const currentLayerSize = network[i];
        const nextLayerSize = network[i + 1];
        
        const x1 = (i + 1) * layerSpacing;
        const x2 = (i + 2) * layerSpacing;
        
        for (let j = 0; j < currentLayerSize; j++) {
            const y1 = ((height - (currentLayerSize * neuronSpacing)) / 2) + (j + 0.5) * neuronSpacing;
            
            for (let k = 0; k < nextLayerSize; k++) {
                const y2 = ((height - (nextLayerSize * neuronSpacing)) / 2) + (k + 0.5) * neuronSpacing;
                
                // Calculate connection weight
                const weight = weights[i][j][k];
                const normalizedWeight = Math.max(-1, Math.min(1, weight));
                
                // Connection color and thickness based on weight
                const connection = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                connection.setAttribute('x1', x1);
                connection.setAttribute('y1', y1);
                connection.setAttribute('x2', x2);
                connection.setAttribute('y2', y2);
                connection.setAttribute('stroke', orangeBlueScale(normalizedWeight));
                connection.setAttribute('stroke-width', Math.abs(normalizedWeight) * 4 + 1);
                connection.setAttribute('opacity', '0.7');
                svg.appendChild(connection);
            }
        }
    }
    
    // Draw neurons
    for (let i = 0; i < network.length; i++) {
        const layerSize = network[i];
        const x = (i + 1) * layerSpacing;
        
        for (let j = 0; j < layerSize; j++) {
            const y = ((height - (layerSize * neuronSpacing)) / 2) + (j + 0.5) * neuronSpacing;
            
            // Neuron value for color
            let neuronValue = 0;
            if (neurons.length > i && neurons[i].length > j) {
                neuronValue = neurons[i][j];
            }
            const normalizedValue = Math.max(-1, Math.min(1, neuronValue));
            
            // Create neuron
            const neuron = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            
            // Neuron circle
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', x);
            circle.setAttribute('cy', y);
            circle.setAttribute('r', neuronRadius);
            circle.setAttribute('fill', '#333');
            circle.setAttribute('stroke', '#666');
            circle.setAttribute('stroke-width', '2');
            neuron.appendChild(circle);
            
            // Neuron value as color in center
            const valueCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            valueCircle.setAttribute('cx', x);
            valueCircle.setAttribute('cy', y);
            valueCircle.setAttribute('r', neuronRadius * 0.7);
            valueCircle.setAttribute('fill', orangeBlueScale(normalizedValue));
            neuron.appendChild(valueCircle);
            
            svg.appendChild(neuron);
            
            // Add labels for input and output layers
            if (i === 0) {
                // Input layer labels
                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('x', x - neuronRadius * 2.2);
                label.setAttribute('y', y + 5);
                label.setAttribute('fill', '#b0b0b0');
                label.setAttribute('font-size', '14');
                label.setAttribute('text-anchor', 'end');
                label.textContent = `X${j+1}`;
                svg.appendChild(label);
            }
        }
    }
    
    // Add layer hints
    for (let i = 0; i < network.length; i++) {
        const layerSize = network[i];
        const x = (i + 1) * layerSpacing;
        const y = height + 20;
        
        // Add neuron count
        if (i > 0 && i < network.length - 1) {
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', x);
            text.setAttribute('y', y);
            text.setAttribute('fill', '#b0b0b0');
            text.setAttribute('font-size', '12');
            text.setAttribute('text-anchor', 'middle');
            text.textContent = `${layerSize} neurons`;
            svg.appendChild(text);
        }
    }
}

function renderOutputVisualization() {
    const container = document.getElementById('output-visualization');
    const canvas = document.createElement('canvas');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    
    // Clear container
    container.innerHTML = '';
    container.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);
    
    // Get data bounds
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    
    for (const point of dataPoints) {
        minX = Math.min(minX, point.x);
        maxX = Math.max(maxX, point.x);
        minY = Math.min(minY, point.y);
        maxY = Math.max(maxY, point.y);
    }
    
    // Add padding to bounds
    const paddingX = (maxX - minX) * 0.1;
    const paddingY = (maxY - minY) * 0.1;
    minX -= paddingX;
    maxX += paddingX;
    minY -= paddingY;
    maxY += paddingY;
    
    // Grid size for visualization
    const gridSize = 50;
    const stepX = (maxX - minX) / gridSize;
    const stepY = (maxY - minY) / gridSize;
    
    // Draw decision boundary
    if (params.features.includes('x1') && params.features.includes('x2')) {
        for (let i = 0; i < gridSize; i++) {
            for (let j = 0; j < gridSize; j++) {
                const x = minX + i * stepX;
                const y = minY + j * stepY;
                
                // Create input vector
                const inputs = getFeatureValues({ x, y });
                
                // Get network prediction
                const prediction = forwardPropagate(inputs);
                
                // Normalize prediction to [0, 1]
                let normalizedPrediction;
                const showDiscrete = document.getElementById('discretize-output').checked;
                
                if (showDiscrete) {
                    normalizedPrediction = prediction > 0.5 ? 1 : 0;
                } else {
                    normalizedPrediction = Math.max(0, Math.min(1, prediction));
                }
                
                // Map to canvas coordinates
                const canvasX = ((x - minX) / (maxX - minX)) * width;
                const canvasY = height - ((y - minY) / (maxY - minY)) * height;
                
                // Draw colored pixel
                const pixelSize = width / gridSize;
                ctx.fillStyle = d3.interpolateRdYlBu(1 - normalizedPrediction);
                ctx.fillRect(canvasX - pixelSize / 2, canvasY - pixelSize / 2, pixelSize, pixelSize);
            }
        }
    }
    
    // Draw data points
    const showTestData = document.getElementById('show-test-data').checked;
    const dataToShow = showTestData ? dataPoints : trainingData;
    
    for (const point of dataToShow) {
        // Map to canvas coordinates
        const canvasX = ((point.x - minX) / (maxX - minX)) * width;
        const canvasY = height - ((point.y - minY) / (maxY - minY)) * height;
        
        // Draw point
        ctx.fillStyle = point.label === 1 ? '#2196f3' : '#ff9800';
        ctx.beginPath();
        ctx.arc(canvasX, canvasY, 4, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw border
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    
    // Draw coordinate axes
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    
    // X-axis
    const xAxisY = height - ((0 - minY) / (maxY - minY)) * height;
    if (xAxisY >= 0 && xAxisY <= height) {
        ctx.beginPath();
        ctx.moveTo(0, xAxisY);
        ctx.lineTo(width, xAxisY);
        ctx.stroke();
    }
    
    // Y-axis
    const yAxisX = ((0 - minX) / (maxX - minX)) * width;
    if (yAxisX >= 0 && yAxisX <= width) {
        ctx.beginPath();
        ctx.moveTo(yAxisX, 0);
        ctx.lineTo(yAxisX, height);
        ctx.stroke();
    }
    
    // Draw grid lines
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    
    // Horizontal grid lines
    for (let i = -6; i <= 6; i++) {
        const y = height - ((i - minY) / (maxY - minY)) * height;
        if (y >= 0 && y <= height) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
            
            // Add label
            ctx.fillStyle = '#999';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(i.toString(), 5, y - 5);
        }
    }
    
    // Vertical grid lines
    for (let i = -6; i <= 6; i++) {
        const x = ((i - minX) / (maxX - minX)) * width;
        if (x >= 0 && x <= width) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
            
            // Add label
            ctx.fillStyle = '#999';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(i.toString(), x, height - 5);
        }
    }
}