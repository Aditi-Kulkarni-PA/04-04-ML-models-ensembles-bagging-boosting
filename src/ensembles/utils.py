"""
Utility functions for data handling and visualization
"""

import numpy as np
from sklearn.model_selection import train_test_split as sklearn_split
from sklearn.datasets import make_classification


def load_sample_data(n_samples=200, n_features=2, n_classes=2, random_state=42):
    """
    Generate a sample classification dataset.
    
    Parameters:
    -----------
    n_samples : int, default=200
        Number of samples.
    n_features : int, default=2
        Number of features.
    n_classes : int, default=2
        Number of classes.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector with labels -1 and 1 for binary classification.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    # Convert labels to -1 and 1 for binary classification
    if n_classes == 2:
        y = np.where(y == 0, -1, 1)
    
    return X, y


def train_test_split(X, y, test_size=0.3, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    test_size : float, default=0.3
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split data.
    """
    return sklearn_split(X, y, test_size=test_size, random_state=random_state)


def compute_metrics(y_true, y_pred):
    """
    Compute basic classification metrics.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
        
    Returns:
    --------
    metrics : dict
        Dictionary containing accuracy, precision, recall, and f1-score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # For binary classification
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == -1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == -1))
    
    # Precision
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    # Recall
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # F1-score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def print_metrics(metrics, model_name="Model"):
    """
    Print classification metrics in a formatted way.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing classification metrics.
    model_name : str, default="Model"
        Name of the model being evaluated.
    """
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
