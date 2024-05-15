<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#Sync-helper-files-from-AWS-S3">Sync helper files from AWS S3</a></li>
    <li><a href="#Load-MNIST-dataset">Load MNIST dataset</a></li>
    <li><a href="#Data-visualization">Data visualization</a></li>
    <li><a href="#Sync-data-from-SageMaker-to-AWS-S3">Sync data from SageMaker to AWS S3</a></li>
    <li><a href="#Data-preparation-for-model-training">Data preparation for model training</a></li>
    <li><a href="#Build-a-Neural-Network-for-solving-binary-MNIST">Build a Neural Network for solving binary MNIST</a></li>
    <li><a href="#Train-the-Neural-Network">Train the Neural Network</a></li>
    <li><a href="#Compute-accuracy-on-train-and-test-datasets">Compute accuracy on train and test datasets</a></li>
    <li><a href="#Visualize-misclassified-elements">Visualize misclassified elements</a></li>
    <li><a href="#summary">Summary</a></li>
  </ol>
</details>

</br>

# Code a Neural Network from scratch to solve the binary MNIST problem

</br>

### Introduction

</br>

This article provides the development of a 3-layer Neural Network (NN) from sratch (i.e., only using Numpy) for solving the binary MNIST dataset. This project offers a practical guide to the foundational aspects of deep learning and the architecture of neural networks. It primarily concentrates on building the network from the ground up (i.e., the mathematics running underthe hood of NNs).

</br>

### Sync helper files from AWS S3

</br>

First of all, we need some helper files from AWS S3. The helper files contain helper functions for data preparation and later constructing the neural network.

-   Uploading helper files to AWS S3;
-   Sync helper files from AWS S3 to AWS SageMaker

</br>

### Load MNIST dataset

</br>

Once the helper files are available in AWS SageMaker, we use pre-defined functions to load the MNIST dataset.

```python
from utils_data import *
download_and_save_MNIST(path="data/")
```

The purpose of this experiment is to handle the binary MNIST only. Therefore, we need a function to load the binary MNIST of 0 and 1 only (i.e., other MNIST digits from 2 to 9 are out of scope in this example).

We applied those functions to load binary MNIST dataset

```python
X_train_org, Y_train_org, X_test_org, Y_test_org = load_mnist()
X_train_org, Y_train_org, X_test_org, Y_test_org = load_binary_mnist(X_train_org, Y_train_org, X_test_org, Y_test_org)
```

</br>

### Data visualization

</br>

```python
visualize_multi_images(X_train_org, Y_train_org, layout=(3, 3), figsize=(10, 10), fontsize=12)
```

<p align="center">
  <a href="">
    <img src="/viz/visual0.png" width="620" alt=""/>
  </a>
</p>

</br>

### Sync data from SageMaker to AWS S3

</br>

I store data in an AWS S3 bucket for later use, if needed.

```python
key = "data/mnist.npz"
bucket_url = "s3://{}/{}".format(BUCKET_NAME, key)
boto3.Session().resource("s3").Bucket(BUCKET_NAME).Object(key).upload_file("data/mnist.npz")
```

</br>

### Data preparation for model training

</br>

We applied a helper function to prepare the binary MNIST for training the Neural Network.

```python
X_train, X_test, Y_train, Y_test = make_inputs(X_train_org, X_test_org, Y_train_org, Y_test_org)
```

</br>

### Build a Neural Network for solving binary MNIST

</br>

To build a Neural Network, we must define helper functions as the building blocks for constructing the architecture. I will not list those functions here because they will make this README unnecessarily long. I only present the construction of the Neural Network. For more details regarding helper functions and components of nn_Llayers_binary(), please see utils_binary.py in this repository and refer to <a href="https://github.com/LeoUtas/2-layer_neural_network.git">2-layer_neural_network</a>.

```python
layer_dims = [784, 128, 64, 1]
learning_rate = 0.01
number_iterations = 250
```

</br>

### Train the Neural Network

</br>

```python
from utils_binary import *

parameters, costs, time = nn_Llayers_binary(X_train, Y_train, layer_dims, learning_rate, number_iterations, print_cost=False)
```

</br>

### Compute accuracy on train and test datasets

</br>

```python
Yhat_train = predict_binary(X_train, Y_train, parameters)
train_accuracy = compute_accuracy(Yhat_train, Y_train)

Yhat_test = predict_binary(X_test, Y_test, parameters)
test_accuracy = compute_accuracy(Yhat_test, Y_test)

print(f"Train accuracy: {train_accuracy} %")
print(f"Test accuracy: {test_accuracy} %")
```

```
Train accuracy: 99.65 %
Test accuracy: 99.81 %
```

Given that the MNIST dataset is not difficult, using only binary MNIST to distinguish between 0 and 1 makes this task even simpler. Therefore, it is no surprise to see such high accuracy on both the train and test datasets, even though the solution presented in this experiment is very simple.

</br>

### Visualize misclassified elements

</br>

<p align="center">
  <a href="">
    <img src="/viz/visual1.png" width="480" alt=""/>
  </a>
</p>

### Summary

This repository could make a great introductory project for those new to artificial intelligence, machine learning, and deep learning. By experimenting with this simple neural network, I learned many basic principles that operate behind the scenes. Also, this example is a beginner-friendly use case of AWS SageMaker and AWS S3 technologies.
