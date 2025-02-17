from enum import Enum
from typing import Union

class ClassificationEvaluationTool(Enum):
    """
    Enum representing evaluation tools for classification tasks.

    Attributes
    ----------
    CONFUSION_MATRIX : str
        Represents the confusion matrix evaluation tool with the name 'CONFUSION_MATRIX'.

    Notes
    -----
    The confusion matrix is used to evaluate the performance of classification models by showing the true versus predicted classifications.
    """
    
    CONFUSION_MATRIX = 'CONFUSION_MATRIX'

class ClassificationMetric(Enum):
    """
    Enum representing metrics for evaluating classification performance.

    Attributes
    ----------
    ACCURACY : str
        Represents the accuracy metric with the name 'ACCURACY'.
    PRECISION : str
        Represents the precision metric with the name 'PRECISION'.
    SPECIFICITY : str
        Represents the specificity metric with the name 'SPECIFICITY'.
    RECALL : str
        Represents the recall metric with the name 'RECALL'.
    FALLOUT : str
        Represents the fallout metric with the name 'FALLOUT'.
    F1_SCORE : str
        Represents the F1 score metric with the name 'F1_SCORE'.

    Notes
    -----
    These metrics are used to assess various aspects of classification model performance.
    """
    
    ACCURACY = 'ACCURACY'
    PRECISION = 'PRECISION'
    SPECIFICITY = 'SPECIFICITY'
    RECALL = 'RECALL'
    FALLOUT = 'FALLOUT'
    F1_SCORE = 'F1_SCORE'

class RegressionEvaluationTool(Enum):
    """
    Enum representing evaluation tools for regression tasks.

    Attributes
    ----------
    RESIDUALS : str
        Represents the residuals evaluation tool with the name 'RESIDUALS'.

    Notes
    -----
    Residuals are used to evaluate the performance of regression models by showing the difference between observed and predicted values.
    """
    
    RESIDUALS = 'RESIDUALS'

class RegressionMetric(Enum):
    """
    Enum representing metrics for evaluating regression performance.

    Attributes
    ----------
    MEAN_ABSOLUTE_ERROR : str
        Represents the mean absolute error metric with the name 'MEAN_ABSOLUTE_ERROR'.
    MEAN_SQUARED_ERROR : str
        Represents the mean squared error metric with the name 'MEAN_SQUARED_ERROR'.

    Notes
    -----
    These metrics are used to measure the error of regression models.
    """
    
    MEAN_ABSOLUTE_ERROR = 'MEAN_ABSOLUTE_ERROR'
    MEAN_SQUARED_ERROR = 'MEAN_SQUARED_ERROR'

Metric = Union[ClassificationMetric, RegressionMetric]
"""
Type alias representing a union of classification and regression metrics.

This type is used to indicate that a metric can be either from classification or regression evaluation.
"""
