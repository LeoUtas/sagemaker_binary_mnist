# load the required packages
import numpy as np
import matplotlib.pyplot as plt
from time import time


# ________________ initialize parameters using for L layer models ________________ #
def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(
            layer_dims[l], layer_dims[l - 1]
        ) / np.sqrt(
            layer_dims[l - 1]
        )  # * 0.01

        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


# ________________ def a sigmoid function ________________ #
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


# ________________ def a ReLU function ________________ #
def relu(Z):
    A = np.maximum(0, Z)

    assert A.shape == Z.shape

    cache = Z
    return A, cache


# ________________ compute the linear forward ________________ #
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    # to check if Z are in the correct shape
    assert Z.shape == (W.shape[0], A.shape[1])
    cache = (A, W, b)

    return Z, cache


# ________________ plug the activation function to the output of linear_forward() ________________ #
def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


# ________________ forward computation of the NN model with L hidden layers ________________ #
def nn_Llayer_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A

        A, cache = linear_activation_forward(
            A_prev,
            parameters["W" + str(l)],
            parameters["b" + str(l)],
            activation="relu",
        )
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid"
    )
    caches.append(cache)

    return AL, caches


# ________________ compute the cost ________________ #
def compute_cost_binary(AL, Y):
    m = Y.shape[1]
    epsilon = 1e-15

    # Compute loss from aL and y.
    cost = (1.0 / m) * (
        -np.dot(Y, np.log(AL + epsilon).T) - np.dot(1 - Y, np.log(1 - AL).T)
    )

    cost = np.squeeze(
        cost
    )  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost


# ________________ compute the linear backward ________________ #
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1.0 / m * np.dot(dZ, A_prev.T)
    db = 1.0 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


# ________________ compute sigmoid backward ________________ #
def sigmoid_backward(dA, cache):
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ


# ________________ compute ReLU backward ________________ #
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, set dz to 0 as well.
    dZ[Z <= 0] = 0

    return dZ


# ________________ compute linear activation backward ________________ #
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


# ________________ backward computation of the NN model with L hidden layers ________________ #
def nn_Llayer_backward_binary(AL, Y, caches):
    grads = {}
    L = len(caches)  # number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # to set Y in the same shape as AL

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]

    (
        grads["dA" + str(L - 1)],
        grads["dW" + str(L)],
        grads["db" + str(L)],
    ) = linear_activation_backward(dAL, current_cache, activation="sigmoid")

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, activation="relu"
        )
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# ________________ update the parameters ________________ #
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = (
            parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        )
        parameters["b" + str(l + 1)] = (
            parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        )

    return parameters


# ________________ prediction ________________ #
def predict_binary(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = nn_Llayer_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    return p


# ________________ compute accuracy ________________ #
def compute_accuracy(p, y):
    m = y.shape[1]
    accuracy = float(np.sum((p == y).astype(int)) / m) * 100
    accuracy = round(accuracy, 2)

    return accuracy


# ________________ plot the costs over iterations ________________ #
def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per hundreds)")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


# ________________ build a L-layer NN model using 1 batch ________________ #
def nn_Llayers_binary(
    X, Y, layer_dims, learning_rate, number_iterations, print_cost=False
):
    start_time = time()

    np.random.seed(1)
    costs = []

    # initialize model parameters
    parameters = initialize_parameters_deep(layer_dims)

    # loop through the iterations
    for i in range(0, number_iterations):
        # forward propagation
        AL, caches = nn_Llayer_forward(X, parameters)

        # compute cost
        cost = compute_cost_binary(AL, Y)

        # backward propagation
        grads = nn_Llayer_backward_binary(AL, Y, caches)

        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # print the cost every 100 iterations
        if print_cost and i % 100 == 0 or print_cost and i == number_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == number_iterations:
            costs.append(cost)

    execution_time = time() - start_time

    print(f"Execution time: {round(execution_time, 2)} seconds")

    return parameters, costs, execution_time
