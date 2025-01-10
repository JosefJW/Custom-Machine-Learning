import unittest
from collections import Counter
from knn.knn import KNN

class TestKNN(unittest.TestCase):

    def setUp(self):
        # Initialize the KNN with k=3
        self.knn = KNN(k=3)
        
        # Sample data and labels for training
        self.data = [
            [1, 2], [2, 3], [3, 3], [6, 5], [7, 8], [8, 7],
            [9, 7], [4, 5], [5, 6], [1, 6]
        ]
        self.labels = [
            "A", "A", "B", "B", "C", "C", "A", "B", "C", "A"
        ]
        
        # Train the KNN model
        self.knn.train(self.data, self.labels)

    def test_train(self):
        """Test that the model trains correctly."""
        self.assertEqual(len(self.knn.data), len(self.data))
        self.assertEqual(len(self.knn.labels), len(self.labels))

    def test_invalid_train(self):
        """Test invalid training due to insufficient data."""
        with self.assertRaises(ValueError):
            self.knn.train(self.data[:1], self.labels[:1])  # Less than k data points
        
        with self.assertRaises(ValueError):
            self.knn.train(self.data, self.labels[:-1])  # Mismatched data and labels

    def test_calculate_distance(self):
        """Test distance calculation."""
        point1 = [1, 2]
        point2 = [2, 3]
        
        # Test Euclidean distance
        self.assertAlmostEqual(self.knn.calculate_distance(point1, point2, "Euclidean"), 1.4142135623730951)
        
        # Test Squared Euclidean distance
        self.assertEqual(self.knn.calculate_distance(point1, point2, "Squared Euclidean"), 2.0)
        
        # Test Manhattan distance
        self.assertEqual(self.knn.calculate_distance(point1, point2, "Manhattan"), 2)

    def test_predict(self):
        """Test prediction functionality."""
        point = [2, 3]
        predicted_label = self.knn.predict(point)
        self.assertIn(predicted_label, ["A", "B", "C"])  # Assuming the model is predicting one of these
        
        # Test invalid input (dimension mismatch)
        with self.assertRaises(ValueError):
            self.knn.predict([1])  # Point with incorrect dimensions

    def test_weighted_predict(self):
        """Test weighted prediction with inverse distance."""
        point = [2, 3]
        predicted_label = self.knn.weighted_predict(point)
        self.assertIn(predicted_label, ["A", "B", "C"])
        
        # Test invalid input (dimension mismatch)
        with self.assertRaises(ValueError):
            self.knn.weighted_predict([1])  # Point with incorrect dimensions

    def test_confidence_predict(self):
        """Test confidence prediction functionality."""
        point = [2, 3]
        confidence = self.knn.confidence_predict(point)
        
        # Ensure confidence values are between 0 and 1 and sum to 1
        if "A" in confidence:
            self.assertGreaterEqual(confidence["A"], 0)
            self.assertLessEqual(confidence["A"], 1)
        if "B" in confidence:
            self.assertGreaterEqual(confidence["B"], 0)
            self.assertLessEqual(confidence["B"], 1)
        if "C" in confidence:
            self.assertGreaterEqual(confidence["C"], 0)
            self.assertLessEqual(confidence["C"], 1)

        # Ensure the sum of confidence values is 1
        self.assertAlmostEqual(sum(confidence.values()), 1)

    def test_invalid_point_in_confidence_predict(self):
        """Test confidence_predict with invalid point input."""
        with self.assertRaises(ValueError):
            self.knn.confidence_predict([1])  # Point with incorrect dimensions

    def test_invalid_distance_type(self):
        """Test invalid distance type."""
        point1 = [1, 2]
        point2 = [2, 3]
        
        with self.assertRaises(ValueError):
            self.knn.calculate_distance(point1, point2, "InvalidDistance")
    
    def test_k_value(self):
        """Test that k is properly initialized."""
        self.assertEqual(self.knn.k, 3)

        # Test invalid k initialization
        with self.assertRaises(ValueError):
            KNN(k=0)  # k must be positive

if __name__ == "__main__":
    unittest.main()
