# Neural Network Visualization

An interactive neural network visualization tool that allows you to experiment with neural networks in real-time.

## Features

- Interactive neural network visualization
- Real-time training with animated epochs
- Multiple dataset options (Circle, Spiral, XOR, Gaussian)
- Customizable network architecture
- Adjustable hyperparameters (learning rate, activation function, regularization)
- Decision boundary visualization
- Feature selection
- Dark-themed UI

## Getting Started

### Prerequisites

- Python 3.6+
- Web browser (Chrome, Firefox, Safari, Edge)

### Installation

No additional libraries are needed to run the application, as it uses standard Python libraries and CDN-hosted JavaScript libraries.

### Running the Application

1. Clone or download this repository
2. Open a terminal/command prompt and navigate to the project directory
3. Run the following command:

```
python server.py
```

4. The application will automatically open in your default web browser
5. If it doesn't open automatically, visit [http://localhost:8000](http://localhost:8000) in your browser

## Usage

### Basic Controls

- **Play/Pause**: Start or pause the training animation
- **Step**: Advance the training by one epoch
- **Reset**: Reset the neural network to its initial state

### Configuration

- **Epoch**: Current training epoch (automatically updates during training)
- **Learning Rate**: Controls how quickly the network learns (0.001 to 1)
- **Activation**: Choose between ReLU, Sigmoid, or Tanh activation functions
- **Regularization**: Apply L1, L2, or no regularization
- **Regularization Rate**: Control the strength of regularization (0 to 1)
- **Problem Type**: Choose between classification or regression

### Data Options

- Select from four different datasets: Circle, Spiral, XOR, or Gaussian
- Adjust the ratio of training to test data
- Add noise to the data
- Set the batch size for training

### Network Architecture

- Add or remove hidden layers
- Each hidden layer has 3 neurons by default
- Network connections are visualized with colors representing weight values
- Neuron colors represent activation values

### Output Visualization

- View the decision boundary as colored regions
- See data points colored by class
- Toggle between continuous and discrete output
- Show or hide test data

## How It Works

The application implements a fully-connected neural network with:

1. Forward propagation to compute predictions
2. Backpropagation with gradient descent to update weights
3. Mini-batch training
4. Various activation functions
5. Regularization options

The visualization is done using HTML Canvas and SVG elements, with D3.js for color scales.

## License

This project is open source and available for educational purposes.

## Acknowledgments

Inspired by various neural network visualization tools and educational platforms.