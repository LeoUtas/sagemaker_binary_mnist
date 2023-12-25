# load the required packages
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import os


# ________________ download and save the MNIST dataset ________________ #
def download_and_save_MNIST(path="data/"):
    # Check if the directory exists, and if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    # Load the MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Save the MNIST data to your local machine
    np.savez(
        os.path.join(path, "mnist.npz"),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )


# ________________ load the MNIST data________________ #
def load_mnist():
    with np.load("data/mnist.npz") as data:
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

    return x_train, y_train, x_test, y_test


# ________________ load the MNIST data for binary classification________________ #
def load_binary_mnist(x_train, y_train, x_test, y_test):
    # Get the indices where y_train and y_test are either 0 or 1
    train_i = np.where((y_train == 0) | (y_train == 1))[0]
    test_i = np.where((y_test == 0) | (y_test == 1))[0]

    # Subset the data based on those indices
    x_train, y_train = x_train[train_i], y_train[train_i]
    x_test, y_test = x_test[test_i], y_test[test_i]

    return x_train, y_train, x_test, y_test


# ________________ visualize one image ________________ #
def visualize_1image(X, Y):
    # data exploration
    X_view = X.reshape(X.shape[0], -1).T
    Y_view = Y

    m, n = X_view.shape
    random_index = np.random.randint(m)
    Xview_random_reshaped = (
        X_view[:, random_index].reshape((28, 28)).T
    )  # Reshape to 28x28

    # plot one image
    plt.imshow(Xview_random_reshaped)
    plt.xlabel(f"Label: {Y_view[random_index]}", fontsize=12)
    plt.title(f"Random index: {random_index}")
    plt.xticks([])
    plt.yticks([])
    plt.show()


# ________________ visualize multiple images ________________ #
def visualize_multi_images(X, Y, layout=(4, 4), figsize=(18, 18), fontsize=28):
    X_view = X.reshape(X.shape[0], -1).T
    Y_view = Y

    m, n = X_view.shape
    rows, cols = layout
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.tight_layout(pad=0.2)

    for i, ax in enumerate(axes.flat):
        # Break the loop if you've reached the end of your dataset
        if i >= n:
            break

        random_index = np.random.randint(n)
        Xview_random_reshaped = X_view[:, random_index].reshape(28, 28).T
        ax.imshow(Xview_random_reshaped)
        ax.set_xlabel(f"Label: {Y_view[random_index]}", fontsize=fontsize)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()  # To display the plot


# ________________ make data ready ________________ #
def make_inputs(X_train_org, X_test_org, Y_train_org, Y_test_org):
    # flatten and normalize the X data
    X_train = (X_train_org / 255.0).reshape(X_train_org.shape[0], -1).T
    X_test = (X_test_org / 255.0).reshape(X_test_org.shape[0], -1).T

    Y_train = Y_train_org.reshape(1, -1)
    Y_test = Y_test_org.reshape(1, -1)

    return X_train, X_test, Y_train, Y_test
