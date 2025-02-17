import numpy as np
from typing import List, Union, Optional
from enums.performance_evaluation_manager_enums import RegressionMetric

class Residuals:
    """
    Class to compute and store residuals for regression evaluation.

    Attributes
    ----------
    residuals : List[float]
        A list to store residuals.
    """

    def __init__(self) -> None:
        """
        Initializes the Residuals object, setting up an empty list for residuals.
        """
        self.residuals = []

    def update_from_divided_source(
        self,
        epoch: int,
        y: Union[float, np.ndarray],
        y_pred: Union[float, np.ndarray]
    ) -> None:
        """
        Updates the residuals list with the difference between true and predicted values.

        Parameters
        ----------
        epoch : int
            The epoch during which the update is performed.
        y : Union[float, np.ndarray]
            True values. Can be a float or a NumPy array.
        y_pred : Union[float, np.ndarray]
            Predicted values. Can be a float or a NumPy array.
        """
        if isinstance(y, float):
            y = np.array([y], dtype=float)

        if isinstance(y_pred, float):
            y_pred = np.array([y_pred], dtype=float)

        incoming_residuals = y - y_pred

        self.residuals.extend(incoming_residuals.tolist())

    def update(self, batch_residuals: np.ndarray) -> None:
        """
        Updates the residuals list with the given batch of residuals.

        Parameters
        ----------
        batch_residuals : np.ndarray
            Array of residuals to be added to the list.
        """
        self.residuals.extend(batch_residuals.tolist())
        
    def __str__(self) -> str:
        """
        Returns a string representation of the Residuals object.
        (Currently returns the default string representation.)

        Returns
        -------
        str
            String representation of the Residuals object.
        """
        return super().__str__()

    def reset(self) -> None:
        """
        Resets the residuals list to an empty list.
        """
        self.residuals = []

    def get_metrics(
        self, 
        evaluation_metrics: Optional[List[RegressionMetric]]
    ) -> dict:
        """
        Computes and returns specified evaluation metrics based on the residuals.

        Parameters
        ----------
        evaluation_metrics : Optional[List[RegressionMetric]]
            List of metrics to compute. If not provided, raises a ValueError.

        Returns
        -------
        dict
            Dictionary of metric names and their computed values.
        """
        if not evaluation_metrics:
            raise ValueError("A list of RegressionMetric is required.")

        residuals_array = np.array(self.residuals)
        return {
            evaluation_metric.value.lower(): getattr(self, evaluation_metric.value.lower())(residuals_array)
            for evaluation_metric in evaluation_metrics
        }

    def mean_absolute_error(self, residuals: np.ndarray) -> float:
        """
        Computes the Mean Absolute Error (MAE) of the given residuals.

        Parameters
        ----------
        residuals : np.ndarray
            Array of residuals.

        Returns
        -------
        float
            Mean Absolute Error of the residuals.
        """
        return np.mean(np.abs(residuals))

    def mean_squared_error(self, residuals: np.ndarray) -> float:
        """
        Computes the Mean Squared Error (MSE) of the given residuals.

        Parameters
        ----------
        residuals : np.ndarray
            Array of residuals.

        Returns
        -------
        float
            Mean Squared Error of the residuals.
        """
        return np.mean(np.square(residuals))