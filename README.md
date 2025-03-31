# Neural Network Training with MATLAB

This MATLAB project implements a basic neural network model with a single hidden layer to classify data points distributed in concentric circles. 
The network is trained using gradient descent with Mean Squared Error (MSE) as the loss function.

## Features
- Generates synthetic data distributed in concentric circles.
- Utilizes a neural network with:
  - **One hidden layer** with a customizable number of neurons.
  - **Sigmoid activation function** for the hidden layer.
  - **Softmax activation function** for the output layer.
- Gradient Descent optimization with learning rate and stopping criterion.
- Visualization of loss progression over epochs.

## Parameters
- `Number_of_neurons`: Number of neurons in the hidden layer (randomly chosen between 10 and 19).
- `epochs`: Number of training iterations (default is 7000).
- `learning_rate`: Learning rate for gradient descent (default is 0.002).
- `stopSIGN`: Stopping criterion threshold for the gradient norm (default is 1e-6).

## Output
- Two plots:
  - **Scatter plot** of the input data points classified into three categories.
  - **Loss curve** showing the training loss progression over the epochs.

## Functions
The script includes the following functions:
- `Sigmoid`: Computes the Sigmoid activation function.
- `Softmax`: Computes the Softmax activation function for the output layer.
- `sigmoid_derivative`: Computes the derivative of the Sigmoid activation function.
  

