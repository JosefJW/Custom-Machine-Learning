import math
import matplotlib.pyplot as plt
from collections import Counter
from numpy import dot
from numpy.linalg import norm

class KNN():
    """
    This class is designed to perform the k-nearest neighbor machine learning algorithm.
    
    On a basic level, the k-nearest neighbor machine learning algorithm takes a point and finds the most common label between its k-nearest neighbors and assigns that label to the point.
    For example, if you have the points [[0, 0], [0, 1], [2, 0], [0, 3], [2, 3], [3, 2]] and the labels [1, 1, 0, 0, 0, 0] and you do k=3-nearest neighbor on the point [1, 1] the algorithm will do the following:
    1) Find the three nearest neighbors of [1, 1] -- they are [0, 0], [0, 1], and [2, 0]
    2) Get the labels for each of the three nearest neighbors -- they are 1, 1, and 0
    3) Choose the most common label -- it is 1
    4) Predict that [1, 1] is that label

    Attributes:
        k (int): The number of nearest neighbors to use in algorithm.
    
    Methods:
        train: Collect data to compare with when making predictions.
        calculate_distance: Calculate the distance between two points
        predict: Predict what class point is based on training data
        weighted_predict: Predict what class point is based on training data using the inverse of the distance as a weight
        confidence_predict: Gives confidence values for classes that point could be
        visualize: Make a graph showing the KNN model
        
    Example:
        Example usage of the class:
        
        >>> data = [[1, 2], [3, 4], [5, 6]]
        >>> labels = [0, 1, 0]
        >>> knn = KNN(k=3)
        >>> knn.train(data, labels)
        >>> prediction_point = [2, 3]
        >>> prediction = knn.predict(prediction_point)
        >>> print(f"Predicted label: {prediction}")
        Predicted label: 0
        
        >>> import random
        >>> data = [
        >>>     [random.randint(1, 500000), random.randint(1, 500000)] for _ in range(100)
        >>> ]
        >>> labels = [random.randint(0, 10) for _ in range(100)]
        >>> knn = KNN(k=30)
        >>> knn.train(data, labels)
        >>> knn.visualize([random.randint(1, 500000), random.randint(1, 500000)], distance_type="Euclidean")
        Shows a visualization of 30NN
        
    """
    
    def __init__(self, k=1):
        """
        Initialize a k-nearest neighbor machine learning object
        
        Args:
            k (int, optional): The number of nearest neighbors to use. Defaults to 1.
            
        Raises:
            ValueError: k must be positive
        """
        if k <= 0:
            raise ValueError("k must be positive.")
        
        self.k = k
    
    def train(self, data, labels):
        """
        Collect data to compare with when making predictions

        Args:
            data (list): List of all points (points described as lists i.e. [x, y, z, ...])
            labels (list): List of labels for each point
        
        Raises:
            ValueError: Not enough training items
            ValueError: Mismatch between data and labels

        Returns:
            KNN: Returns trained model
        """
        if len(data) < self.k:
            raise ValueError(f"Not enough training items for {self.k}-nearest neighbor training.")
        if len(data) != len(labels):
            raise ValueError(f"Data and label lengths do not match. Data: {len(data)} Labels: {len(labels)}")
        
        self.data = data
        self.labels = labels
        return self
    
    def calculate_distance(self, point1, point2, distance_type="Euclidean"):
        """
        Calculate the distance between two points

        Args:
            point1 (list): Coordinates of point1
            point2 (list): Coordinates of point2
            distance_type (str, optional): Type of distance measurement to use. Defaults to "Euclidean".

        Raises:
            ValueError: Points have different dimensions

        Returns:
            int: Distance between the two points
        """
        if not (len(point1) == len(point2)):
            raise ValueError(f"Points have different dimensions.")
        if distance_type not in ["Euclidean", "Squared Euclidean", "Manhattan", 
                                 "Cosine", "Chebyshev", "Hamming", "Jaccard",
                                 "Bray-Curtis", "Canberra"]:
            raise ValueError(f"Unsupported distance type: {distance_type}")
        
        distance = 0
        if distance_type == "Euclidean":
            distance = math.sqrt(sum((p1-p2)**2 for p1, p2 in zip(point1, point2)))
        elif distance_type == "Squared Euclidean":
            distance = sum((p1-p2)**2 for p1, p2 in zip(point1, point2))
        elif distance_type == "Manhattan":
            distance = sum(abs(p1-p2) for p1, p2 in zip(point1, point2))
        elif distance_type == "Cosine":
            if norm(point1) == 0 or norm(point2) == 0:
                distance = 1
            else:
                cos_sim = float(dot(point1, point2)) / (norm(point1)*norm(point2))
                distance = 1 - cos_sim
        elif distance_type == "Chebyshev":
            distance = max(abs(p1-p2) for p1, p2 in zip(point1, point2))
        elif distance_type == "Hamming":
            distance = sum(int(p1 == p2) for p1, p2 in zip(point1, point2))
        elif distance_type == "Jaccard":
            if len(set(point1)) == 0 and len(set(point2)) == 0:
                distance = 1
            else:
                intersection = len(list(set(point1).intersection(point2)))
                union = (len(set(point1)) + len(set(point2))) - intersection
                jaccard_sim = float(intersection) / union
                distance = 1 - jaccard_sim
        elif distance_type == "Bray-Curtis":
            sum1 = sum(abs(p1-p2) for p1, p2 in zip(point1, point2))
            sum2 = sum(p1+p2 for p1, p2 in zip(point1, point2))
            distance = float(sum1)/sum2
        elif distance_type == "Canberra":
            distance = sum(float(abs(p1-p2))/(abs(p1)+abs(p2)) for p1, p2 in zip(point1, point2))
        
        return distance
                
    def predict(self, point, distance_type="Euclidean"):
        """
        Predict what class point is based on training data

        Args:
            point (list): Coordinates of point
            distance_type (str, optional): Type of distance measurement to use. Defaults to "Euclidean".
        
        Raises:
            ValueError: Points have different dimensions
        
        Returns:
            label: Predicted label
        """
        if not isinstance(point, list) or not (len(point) == len(self.data[0])):
            raise ValueError("Invalid point or dimension mismatch with training data.")
        
        distances = [(self.calculate_distance(point, datapoint, distance_type), label) for datapoint, label in zip(self.data, self.labels)]
        distances.sort(key=lambda x: x[0])  # Sort by distance
        nearest_neighbors = distances[:self.k]  # Take the k nearest neighbors

        # Count the occurrences of each label
        labels = [label for _, label in nearest_neighbors]
        counter = Counter(labels)
        return counter.most_common(1)[0][0] 
    
    def weighted_predict(self, point, distance_type="Euclidean"):
        """
        Predict what class point is based on training data using the inverse of the distance as a weight

        Args:
            point (list): Coordinates of point
            distance_type (str, optional): Type of distance measurement to use. Defaults to "Euclidean".
        
        Raises:
            ValueError: Points have different dimensions
        
        Returns:
            label: Predicted label
        """
        if not isinstance(point, list) or not (len(point) == len(self.data[0])):
            raise ValueError("Invalid point or dimension mismatch with training data.")
        
        distances = [(self.calculate_distance(point, datapoint, distance_type), label) for datapoint, label in zip(self.data, self.labels)]
        distances.sort(key=lambda x: x[0])  # Sort by distance
        nearest_neighbors = distances[:self.k]  # Take the k nearest neighbors

        labels = {}
        for distance, label in nearest_neighbors:
            if distance == 0:
                labels[label] = float('inf')
            else:
                if label in labels:
                    labels[label] += 1/distance
                else:
                    labels[label] = 1/distance
          
        return max(labels, key=labels.get) 
    
    def confidence_predict(self, point, distance_type="Euclidean"):
        """
        Gives confidence values for classes that point could be

        Args:
            point (list): Coordinates of point
            distance_type (str, optional): Type of distance measurement to use. Defaults to "Euclidean".
        
        Raises:
            ValueError: Points have different dimensions
        
        Returns:
            label: Predicted label confidence values
        """
        
        if not isinstance(point, list) or not (len(point) == len(self.data[0])):
            raise ValueError("Invalid point or dimension mismatch with training data.")
        
        distances = [(self.calculate_distance(point, datapoint, distance_type), label) for datapoint, label in zip(self.data, self.labels)]
        distances.sort(key=lambda x: x[0])  # Sort by distance
        nearest_neighbors = distances[:self.k]  # Take the k nearest neighbors

        # Count the occurrences of each label
        labels = [label for _, label in nearest_neighbors]
        counter = Counter(labels)
        confidence_values = {label: count / self.k for label, count in counter.items()}
        return confidence_values
    
    def visualize(self, point, distance_type="Euclidean"):
        """
        Visualizes the training data, the query point, its predicted label,
        and the k nearest neighbors with lines drawn to them.
        
        Args:
            point (list): Coordinates of the query point for prediction
            distance_type (str, optional): Type of distance measurement to use. Defaults to "Euclidean".
        """
        # Perform prediction for the given point
        predicted_label = self.predict(point, distance_type)
        
        # Calculate distances and find k nearest neighbors
        distances = [(self.calculate_distance(point, datapoint, distance_type), label, datapoint) 
                    for datapoint, label in zip(self.data, self.labels)]
        distances.sort(key=lambda x: x[0])  # Sort by distance
        nearest_neighbors = distances[:self.k]  # Take the k nearest neighbors
        
        # Plot training data points
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            [x[0] for x in self.data],  # x coordinates of training data
            [x[1] for x in self.data],  # y coordinates of training data
            c=self.labels,  # Color points based on their labels
            cmap=plt.cm.Paired,
            s=100,
            marker='o'
        )
        
        # Add color legend for labels
        plt.legend(handles=scatter.legend_elements()[0], labels=set(self.labels), title="Classes")
        
        # Plot the query point
        plt.scatter(point[0], point[1], color='black', s=100, label="Query Point", edgecolor='white', marker='X')
        
        # Draw lines from query point to its k nearest neighbors
        for _, _, neighbor in nearest_neighbors:
            plt.plot([point[0], neighbor[0]], [point[1], neighbor[1]], color='gray', linestyle='dashdot', alpha=0.3)
        
        # Display the predicted label
        plt.title(f"{self.k}NN Prediction: {predicted_label}")
        
        # Show plot
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        plt.show()