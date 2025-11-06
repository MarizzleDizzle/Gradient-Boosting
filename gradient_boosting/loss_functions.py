"""
Loss functions for Gradient Boosting
"""
import numpy as np


class SquaredLoss:
    """Squared loss for regression problems."""

    def __call__(self, y_true, y_pred):
        """
        Calculate the squared loss.

        Parameters:
        -----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted values

        Returns:
        --------
        float : Mean squared error
        """
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, y_true, y_pred):
        """
        Calculate the negative gradient (residuals).

        Parameters:
        -----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted values

        Returns:
        --------
        array : Negative gradient (residuals)
        """
        return y_true - y_pred

    def init_estimate(self, y):
        """
        Initial estimate for the model.

        Parameters:
        -----------
        y : array-like
            Target values

        Returns:
        --------
        float : Mean of target values
        """
        return np.mean(y)


class LogLoss:
    """Log loss for binary classification problems."""

    def __call__(self, y_true, y_pred):
        """
        Calculate the log loss.

        Parameters:
        -----------
        y_true : array-like
            True target values (0 or 1)
        y_pred : array-like
            Predicted probabilities

        Returns:
        --------
        float : Log loss
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, y_true, y_pred):
        """
        Calculate the negative gradient.

        Parameters:
        -----------
        y_true : array-like
            True target values (0 or 1)
        y_pred : array-like
            Predicted probabilities

        Returns:
        --------
        array : Negative gradient
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return y_true - y_pred

    def init_estimate(self, y):
        """
        Initial estimate for the model.

        Parameters:
        -----------
        y : array-like
            Target values (0 or 1)

        Returns:
        --------
        float : Log-odds of the mean
        """
        mean = np.mean(y)
        mean = np.clip(mean, 1e-15, 1 - 1e-15)
        return np.log(mean / (1 - mean))

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

