import numpy as np

class LinearRegression():
    """
    This class implements a machine learning model for linear regression.
    It provides methods to train the model on data, make predictions on new data, and evaluate the performance
    using common metrics like Mean Squared Error (MSE) and R-squared (R^2).

    Attributes:
        B (numpy.ndarray): Coefficients/parameters of the trained model (i.e. the slopes and intercept).
    
    Methods:
        __init__():
            Initializes the model.

        fit(X, y):
            Trains the model by fitting it to the provided training data (X and y).
            A linear regression model is of the form: B_0 + B_1 * X_1 + B_2 * X_2 + ...
            To fit a linear regression model, you just calculate those B values.
            A matrix containing all the B values can be found by solving the following equation:
            B = (X.T @ X)^-1 @ X.T @ y
            
        predict(X):
            Makes predictions on new data based on the trained model.
            Plugs the X values into the regression model B_0 + B_1 * X_1 + B_2 * X_2 + ...

        compute_loss(y_true, y_pred):
            Computes the loss metric to evaluate the model's performance.
            MSE can be anywhere from 0 to infinity.
            MSE measures how far the true y-values are from the regression function.
            An MSE close to 0 means that the model's predictions exactly match the true y-vales.
            MSE = 1/n * sum((y_true-y_pred)^2)

        score(y_true, y_pred):
            Computes the R^2 score to assess how well the model explains the variance in the target variable.
            R^2 values can be between 0 and 1.
            An R^2 close to 0 means the model explains very little about the true value's variance.
            An R^2 close to 1 means the model explains the variance in y very well.
            R^2 = 1 - sum((y_true - y_pred)^2) / sum((y_true - y_mean)^2)
    
    Example:
        Example usage of the class:
        
        >>> X = [[1, 2], [3, 4], [5, 6]]
        >>> y = [0, 1, 0]
        >>> LR = LinearRegression()
        >>> B = LR.fit(X, y)
        >>> prediction_point = [2, 3]
        >>> prediction = LR.predict(prediction_point)
        >>> print(f"Predicted value: {prediction[0]}")
        Predicted value: 0.333333333333324
    """
    
    
    def __init__(self):
        self.B = None
    
    def fit(self, X, y):
        """
        Make the linear regression model's function.
        A linear regression model is of the form: B_0 + B_1 * X_1 + B_2 * X_2 + ...
        To fit a linear regression model, you just calculate those B values.
        A matrix containing all the B values can be found by solving the following equation:
        B = (X.T @ X)^-1 @ X.T @ y

        Args:
            X (list): List of input lists
            y (list): List of outputs

        Returns:
            float list: List of model coefficients
        """
        y_mat = np.array(y).reshape(-1, 1)
        
        X_mat = np.array(X)
        ones_column = np.ones((X_mat.shape[0], 1))
        X_mat = np.hstack((ones_column, X_mat))
        
        B = np.linalg.pinv(X_mat.T @ X_mat) @ X_mat.T @ y_mat
        self.B = B.flatten()
        return self.B
    
    def predict(self, X):
        """
        Plugs inputs into the model's function.
        Plugs the X values into the regression model B_0 + B_1 * X_1 + B_2 * X_2 + ...
        
        Args:
            X (list): List of inputs

        Returns:
            float list: Output from plugging each x into the function
        """
        if self.B is None:
            raise ValueError("B values not calculated yet")

        if isinstance(X[0], list):
            predictions = []
            for i in range(len(X)):
                y = self.B[0]
                for j in range(len(X[i])):
                    y += X[i][j] * self.B[j+1]
                predictions.append(y)
            return predictions
        else:
            y = self.B[0]
            for i in range(len(X)):
                y += X[i]*self.B[i+1]
            return [y]
    
    def compute_loss(self, y_true, y_pred):
        """
        Computes the Mean Squared Error for the regression line.
        MSE can be anywhere from 0 to infinity.
        MSE measures how far the true y-values are from the regression function.
        An MSE close to 0 means that the model's predictions exactly match the true y-vales.
        
        MSE = 1/n * sum((y_true-y_pred)^2)

        Args:
            y_true (list): Target values
            y_pred (list): Predicted values

        Returns:
            float: Mean Squared Error for the model
        """
        if len(y_true) != len(y_pred) or len(y_pred) == 0:
            raise ValueError("Array length error.")
        
        MSE = sum((yt-yp)**2 for yt, yp in zip(y_true, y_pred)) / len(y_pred)
        return MSE
    
    def score(self, y_true, y_pred):
        """
        Computes the R^2 score to assess the model's fit.
        R^2 values can be between 0 and 1.
        An R^2 close to 0 means the model explains very little about the true value's variance.
        An R^2 close to 1 means the model explains the variance in y very well.
        
        R^2 = 1 - sum((y_true - y_pred)^2) / sum((y_true - y_mean)^2)

        Args:
            y_true (list): Target values
            y_pred (list): Predicted values

        Returns:
            float: R^2 score of the model
        """
        mean = sum(y_true)/len(y_true)
        R2 = 1 - sum((yt-yp)**2 for yt, yp in zip(y_true, y_pred)) / sum((yt-mean)**2 for yt in y_true)
        return R2