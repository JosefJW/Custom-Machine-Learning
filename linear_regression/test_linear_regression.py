import unittest
import numpy as np
from linear_regression.linear_regression import LinearRegression

class TestLinearRegression(unittest.TestCase):

    def test_fit_and_predict(self):
        # Sample data for training
        X = [[1, 2], [3, 4], [5, 6]]
        y = [0, 1, 0]

        # Create LinearRegression instance and fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Check if coefficients are calculated
        self.assertIsNotNone(model.B)
        
        # Test prediction for a known point
        prediction = model.predict([2, 3])
        expected_prediction = [0.333333333333324] 
        self.assertAlmostEqual(prediction[0], expected_prediction[0], places=6)

    def test_compute_loss(self):
        # Sample true and predicted values
        y_true = [0, 1, 0]
        y_pred = [0.2, 0.8, 0.1]
        
        # Create LinearRegression instance
        model = LinearRegression()
        
        # Compute MSE (Mean Squared Error)
        mse = model.compute_loss(y_true, y_pred)
        
        # Expected MSE calculation
        expected_mse = 0.03 
        
        self.assertAlmostEqual(mse, expected_mse, places=6)

    def test_score(self):
        # Sample true and predicted values
        y_true = [0, 1, 0]
        y_pred = [0.2, 0.8, 0.1]
        
        # Create LinearRegression instance
        model = LinearRegression()
        
        # Compute R^2 score
        r2_score = model.score(y_true, y_pred)
        
        # Expected R^2 calculation
        expected_r2 = 0.865
        
        self.assertAlmostEqual(r2_score, expected_r2, places=6)

    def test_predict_before_fit(self):
        model = LinearRegression()
        
        # Test if predict raises error when fit has not been called
        with self.assertRaises(ValueError):
            model.predict([2, 3])

    def test_mismatched_length(self):
        # Sample true and predicted values with mismatched lengths
        y_true = [0, 1, 0]
        y_pred = [0.2, 0.8]
        
        # Create LinearRegression instance
        model = LinearRegression()
        
        # Test if compute_loss raises an error for mismatched lengths
        with self.assertRaises(ValueError):
            model.compute_loss(y_true, y_pred)

if __name__ == "__main__":
    unittest.main()
