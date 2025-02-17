from typing import List, Union, Optional
import numpy as np

from enums.performance_evaluation_manager_enums import ClassificationMetric

class ConfusionMatrix:
    """
    A confusion matrix for evaluating classification performance.

    This class tracks the confusion matrix, allows updates based on true and predicted labels or indices, 
    and provides metrics such as accuracy, precision, recall, and more.

    Parameters
    ----------
    class_labels : List[str]
        List of class labels for the confusion matrix. The order determines the indexing in the matrix.

    Attributes
    ----------
    class_labels : List[str]
        List of class labels for the confusion matrix.
    n_classes : int
        Number of classes, derived from `class_labels`.
    label_to_index : dict
        A dictionary mapping class labels to their respective indices.
    matrix : np.ndarray
        The confusion matrix, initialized to zero.

    Methods
    -------
    update(true_labels_or_indices, predicted_labels_or_indices, is_index=False):
        Updates the confusion matrix with true and predicted labels or indices.
    get_matrix():
        Returns the current confusion matrix.
    add(other):
        Adds another confusion matrix to the current one.
    reset():
        Resets the confusion matrix to zeros.
    get_metrics(evaluation_metrics: Optional[List[ClassificationMetric]]):
        Computes specified classification metrics based on the confusion matrix.
    accuracy():
        Computes the accuracy metric.
    precision():
        Computes the precision metric for each class.
    specificity():
        Computes the specificity (True Negative Rate) metric for each class.
    recall():
        Computes the recall metric for each class.
    fallout():
        Computes the fallout (False Positive Rate) metric for each class.
    f1_score():
        Computes the F1 score metric for each class.
    """
    def __init__(self, class_labels: List[str]) -> None:
        self.class_labels = class_labels
        self.n_classes = len(class_labels)
        self.label_to_index = {label: idx for idx, label in enumerate(class_labels)}
        self.matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)

    def _validate_labels(self, labels: List[str], label_type: str):
        """
        Validates the provided labels against the class labels.

        Parameters
        ----------
        labels : List[str]
            List of labels to validate.
        label_type : str
            Type of labels (e.g., 'true', 'predicted') for error messaging.

        Raises
        ------
        ValueError
            If any of the labels are not in the class labels.
        """
        invalid_labels = set(labels) - set(self.class_labels)
        if invalid_labels:
            raise ValueError(f"Invalid class labels for {label_type} label(s): {invalid_labels}")

    def _validate_indices(self, indices: List[int], label_type: str):
        """
        Validates the provided indices.

        Parameters
        ----------
        indices : List[int]
            List of indices to validate.
        label_type : str
            Type of indices (e.g., 'true', 'predicted') for error messaging.

        Raises
        ------
        ValueError
            If any of the indices are out of range.
        """
        if not all(0 <= idx < self.n_classes for idx in indices):
            raise ValueError(f"Invalid indices for {label_type} label(s). Indices must be between 0 and {self.n_classes - 1}")

    def update(
        self,
        # matrix: np.ndarray,
        true_labels_or_indices: Union[str, List[Union[str, int]]],
        predicted_labels_or_indices: Union[str, List[Union[str, int]]],
        is_index: bool = False
    ):
        """
        Updates the confusion matrix with true and predicted labels or indices.

        Parameters
        ----------
        true_labels_or_indices : Union[str, List[Union[str, int]]]
            True labels or indices. Can be a single label/ index or a list of them.
        predicted_labels_or_indices : Union[str, List[Union[str, int]]]
            Predicted labels or indices. Can be a single label/ index or a list of them.
        is_index : bool, optional
            If True, the labels are interpreted as indices. Defaults to False.

        Raises
        ------
        ValueError
            If the number of true and predicted labels (or indices) do not match, or if labels or indices are invalid.
        """
        if isinstance(true_labels_or_indices, str):
            true_labels_or_indices = [true_labels_or_indices]

        if isinstance(predicted_labels_or_indices, str):
            predicted_labels_or_indices = [predicted_labels_or_indices]

        if len(true_labels_or_indices) != len(predicted_labels_or_indices):
            raise ValueError("The number of true labels (indices) and predicted labels (indices) must be the same")

        if is_index:
            true_indices = true_labels_or_indices
            predicted_indices = predicted_labels_or_indices
            self._validate_indices(true_indices, 'true')
            self._validate_indices(predicted_indices, 'predicted')
        else:
            self._validate_labels(true_labels_or_indices, 'true')
            self._validate_labels(predicted_labels_or_indices, 'predicted')
            true_indices = [self.label_to_index[label] for label in true_labels_or_indices]
            predicted_indices = [self.label_to_index[label] for label in predicted_labels_or_indices]

        np.add.at(self.matrix, (true_indices, predicted_indices), 1)

    def get_matrix(self):
        """
        Returns the current confusion matrix.

        Returns
        -------
        np.ndarray
            The confusion matrix.
        """
        return self.matrix

    def add(self, other):
        """
        Adds another confusion matrix to the current one.

        Parameters
        ----------
        other : ConfusionMatrix
            Another ConfusionMatrix instance to add.

        Raises
        ------
        ValueError
            If the class labels of the two matrices do not match.
        """
        if self.class_labels != other.class_labels:
            raise ValueError("Class labels must match to add confusion matrices")
        self.matrix += other.get_matrix()

    def _get_matrix_str_representation(self):
        str_repr = "Confusion Matrix:\n"
        str_repr += "\t\tPredicted\n"
        str_repr += "\t\t" + "\t".join(self.class_labels) + "\n"
        for i, label in enumerate(self.class_labels):
            str_repr += f"Actual {label}\t" + "\t".join(map(str, self.matrix[i])) + "\n"
        return str_repr
    
    def __str__(self):
        str_repr = "Confusion Matrix:\n"
        str_repr += "\t\tPredicted\n"
        str_repr += "\t\t" + "\t".join(self.class_labels) + "\n"
        for i, label in enumerate(self.class_labels):
            str_repr += f"Actual {label}\t" + "\t".join(map(str, self.matrix[i])) + "\n"
        return str_repr
    
    def reset(self):
        """
        Resets the confusion matrix to zeros.
        """
        self.matrix = np.zeros_like(self.matrix)

    def get_metrics(self, evaluation_metrics: Optional[List[ClassificationMetric]]):
        """
        Computes specified classification metrics based on the confusion matrix.

        Parameters
        ----------
        evaluation_metrics : Optional[List[ClassificationMetric]]
            List of ClassificationMetric enums to compute. If None, an exception is raised.

        Returns
        -------
        dict
            A dictionary of computed metrics where keys are metric names and values are metric values.

        Raises
        ------
        ValueError
            If no evaluation metrics are provided.
        """
        if not evaluation_metrics:
            raise ValueError("Need a list of ClassificationMetric to proceed")

        return {
            evaluation_metric.value.lower(): getattr(self, evaluation_metric.value.lower())()
            for evaluation_metric in evaluation_metrics
        }
    
    def accuracy(self):
        """
        Computes the accuracy of the classification.

        Returns
        -------
        float
            The accuracy of the classification, calculated as the ratio of correct predictions to total predictions.
        """
        correct = np.trace(self.matrix)
        total = np.sum(self.matrix)
        return correct / total if total > 0 else 0

    def precision(self):
        """
        Computes the precision for each class.

        Returns
        -------
        dict
            A dictionary where keys are class labels and values are precision scores for each class.
        """
        tp = np.diag(self.matrix)
        fp = np.sum(self.matrix, axis=0) - tp
        precision_scores = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        return dict(zip(self.class_labels, precision_scores))

    def specificity(self):
        """
        Computes the specificity (True Negative Rate) for each class.

        Returns
        -------
        dict
            A dictionary where keys are class labels and values are specificity scores for each class.
        """
        tn = np.sum(self.matrix) - (np.sum(self.matrix, axis=0) + np.sum(self.matrix, axis=1) - np.diag(self.matrix))
        fp = np.sum(self.matrix, axis=0) - np.diag(self.matrix)
        specificity_scores = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0)
        return dict(zip(self.class_labels, specificity_scores))

    def recall(self):
        """
        Computes the recall for each class.

        Returns
        -------
        dict
            A dictionary where keys are class labels and values are recall scores for each class.
        """
        tp = np.diag(self.matrix)
        fn = np.sum(self.matrix, axis=1) - tp
        recall_scores = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        return dict(zip(self.class_labels, recall_scores))

    def fallout(self):
        """
        Computes the fallout (False Positive Rate) for each class.

        Returns
        -------
        dict
            A dictionary where keys are class labels and values are fallout scores for each class.
        """
        fp = np.sum(self.matrix, axis=0) - np.diag(self.matrix)
        tn = np.sum(self.matrix) - (np.sum(self.matrix, axis=0) + np.sum(self.matrix, axis=1) - np.diag(self.matrix))
        fallout_scores = np.divide(fp, fp + tn, out=np.zeros_like(fp, dtype=float), where=(fp + tn) != 0)
        return dict(zip(self.class_labels, fallout_scores))

    def f1_score(self):
        """
        Computes the F1 score for each class.

        Returns
        -------
        dict
            A dictionary where keys are class labels and values are F1 scores for each class.
        """
        precision = self.precision()
        recall = self.recall()
        f1_scores = {
            class_label: 2 * precision[class_label] * recall[class_label] / (precision[class_label] + recall[class_label])
            if (precision[class_label] + recall[class_label]) > 0 else 0
            for class_label in self.class_labels
        }
        return f1_scores
    

    def __str__(self):
        """
        Returns a string representation of the confusion matrix.

        Returns
        -------
        str
            A formatted string showing the confusion matrix.
        """
        str_repr = "Confusion Matrix:\n"
        str_repr += "\t\tPredicted\n"
        str_repr += "\t\t" + "\t".join(self.class_labels) + "\n"
        for i, label in enumerate(self.class_labels):
            str_repr += f"Actual {label}\t" + "\t".join(map(str, self.matrix[i])) + "\n"
        return str_repr