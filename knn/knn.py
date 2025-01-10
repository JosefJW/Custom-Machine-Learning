import math
from collections import Counter
from numpy import dot
from numpy.linalg import norm

class KNN():
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
