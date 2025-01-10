import numpy as np
import matplotlib.pyplot as plt
import random

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
            
        visualize():
            Generates a 2D or 3D linear regression graph based on the data
    
    Example:
        Example usage of the class:
        
        >>> X = [[1, 2], [3, 4], [5, 6]]
        >>> y = [0, 1, 0]
        >>> lr = LinearRegression()
        >>> B = lr.fit(X, y)
        >>> prediction_point = [2, 3]
        >>> prediction = lr.predict(prediction_point)
        >>> print(f"Predicted value: {prediction[0]}")
        Predicted value: 0.333333333333324
        
        >>> X_2d = [[random.randint(1, 10)] for _ in range(5)]
        >>> y_2d = [random.randint(1, 20) for _ in range(5)]
        >>> lr = LinearRegression()
        >>> lr.fit(X_2d, y_2d)
        >>> lr.visualize(X_2d, y_2d, prediction_point=[random.randint(1, 10)])
        >>> X_3d = np.array([[random.randint(1, 10), random.randint(1, 10)] for _ in range(5)])  # 5 rows, 2 columns of random numbers
        >>> y_3d = [random.randint(1, 20) for _ in range(5)]  # List of 5 random numbers between 1 and 20
        >>> lr.fit(X_3d, y_3d)
        >>> lr.visualize(X_3d, y_3d, prediction_point=[random.randint(1, 10), random.randint(1, 10)])
        Makes a 2D linear regression graph and then a 3D linear regression graph
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

        print(isinstance(X[0], list))
        print(X)
        if isinstance(X[0], list) or isinstance(X[0], np.ndarray):
            predictions = []
            for i in range(len(X)):
                y = self.B[0]
                for j in range(len(X[i])):
                    y += X[i][j] * self.B[j+1]
                predictions.append(y)
            return np.array(predictions)
        else:
            y = self.B[0]
            for i in range(len(X)):
                y += X[i]*self.B[i+1]
            return np.array([y])
    
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
    
    def visualize(self, X, y, prediction_point=None):
        """
        Visualizes the training data and the regression model in 2D or 3D.

        Args:
            X (list or np.ndarray): Training input data (2D list or array).
            y (list or np.ndarray): Target output data.
            prediction_point (list, optional): A point to make a prediction on, if provided.
        """
        if len(X[0]) == 1: #2D graph
            plt.figure(figsize=(8, 6))

            # Plot the training data
            plt.scatter(X, y, color='blue', label='Training Data', zorder=5)

            # Fit the model
            self.fit(X, y)

            # Make predictions over a range of x values for plotting the regression line
            x_range = np.linspace(min([x[0] for x in X]) - 1, max([x[0] for x in X]) + 1, 100).reshape(-1, 1)  # Extract the first element of each sublist
            y_pred = self.predict(x_range)

            # Plot the regression line
            plt.plot(x_range, y_pred, color='red', label='Regression Line', zorder=10)

            # If a prediction point is provided, predict its value and plot it
            if prediction_point:
                prediction = self.predict([prediction_point])
                plt.scatter(prediction_point[0], prediction[0], color='green', s=100, label='Prediction Point', zorder=15)

            # Customize plot
            plt.xlabel('X')
            plt.ylabel('y')
            plt.title('2D Linear Regression Visualization')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        elif len(X[0]) == 2: # 3D graph
            from mpl_toolkits.mplot3d import Axes3D

            plt.figure(figsize=(10, 8))
            ax = plt.axes(projection='3d')

            # Plot the training data
            ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Training Data')

            # Fit the model
            self.fit(X, y)

            # Create a meshgrid to visualize the regression plane
            x_range = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 100)  # Extend beyond the min and max for x
            y_range = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1, 100)  # Extend beyond the min and max for y
            X_range, Y_range = np.meshgrid(x_range, y_range)
            Z_range = self.predict(np.c_[X_range.ravel(), Y_range.ravel()]).reshape(X_range.shape)

            # Plot the regression plane
            ax.plot_surface(X_range, Y_range, Z_range, color='red', alpha=0.5)

            # If a prediction point is provided, predict its value and plot it
            if prediction_point:
                prediction = self.predict([prediction_point])
                ax.scatter(prediction_point[0], prediction_point[1], prediction[0], color='green', s=100, label='Prediction Point')

            # Customize plot
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('y')
            ax.set_title('Linear Regression Visualization in 3D')
            ax.legend()
            plt.show()
        else:
            print("Visualization supports only 1D and 2D input vectors.")
