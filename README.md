Custom Machine Learning

Welcome to the Custom Machine Learning repository! This project is an ongoing effort to implement machine learning algorithms from scratch in Python. The goal of this repository is to build various fundamental ML algorithms and showcase the underlying principles behind them. It’s designed for learning, experimentation, and improving understanding of how machine learning works at a low level.
Table of Contents

Overview

This repository contains implementations of various machine learning algorithms, written from scratch without the use of high-level libraries like Scikit-learn or TensorFlow. The aim is to help learners, students, and developers gain a deeper understanding of how these algorithms work by exploring their core mechanics.
Why Build ML Algorithms from Scratch?

Building machine learning models from scratch allows you to:

    Understand the mathematical principles behind each algorithm.
    Gain insights into algorithm design and optimization techniques.
    Strengthen your Python programming and problem-solving skills.

Algorithms Implemented

Here are some of the machine learning algorithms that have been or will be implemented in this repository:

    Supervised Learning:
        K-Nearest Neighbors (KNN)
        Linear Regression
        Logistic Regression
        Decision Trees
        Random Forest

    Unsupervised Learning:
        K-Means Clustering
        Hierarchical Clustering
        Principal Component Analysis (PCA)

    Reinforcement Learning:
        Q-Learning

    Optimization Algorithms:
        Gradient Descent
        Stochastic Gradient Descent (SGD)

Getting Started

To get started with using the algorithms in this repository, follow the steps below.
Prerequisites

Make sure you have Python 3.x installed. You'll also need to install a few dependencies that can be found in requirements.txt. You can install them by running:

pip install -r requirements.txt

Clone the Repository

To clone this repository, run the following command in your terminal:

git clone https://github.com/josefjw/custom-machine-learning.git
cd custom-machine-learning

Usage

Each algorithm is implemented in its own Python file. Here's an example of how to use the K-Nearest Neighbors (KNN) algorithm:
Example: K-Nearest Neighbors (KNN)

from knn.knn import KNN

# Training data: (features, labels)
data = [[1, 2], [3, 4], [5, 6]]
labels = [0, 1, 0]

# Create a KNN model
model = KNN(k=3)

# Train the model
model.train(data, labels)

# Make a prediction
point = [2, 3]
prediction = model.predict(point)
print(f"Predicted label: {prediction}")

Each algorithm will come with similar instructions and examples on how to use it in your own projects.
Contributing

Contributions to this project are welcome! If you have suggestions, improvements, or want to add new algorithms, feel free to open an issue or create a pull request.
How to contribute:

    Fork the repository.
    Create a new branch (git checkout -b feature-name).
    Implement the feature or fix the bug.
    Run the tests to make sure everything works (python -m unittest).
    Commit your changes (git commit -am 'Add feature').
    Push to the branch (git push origin feature-name).
    Open a pull request to merge your changes into the main branch.

License

This project is licensed under the MIT License - see the LICENSE file for details.
