"""
Bagging (Bootstrap Aggregating) Classifier Implementation
"""

import numpy as np
from collections import Counter


class BaggingClassifier:
    """
    Bagging (Bootstrap Aggregating) classifier.
    
    Bagging is an ensemble method that trains multiple models on different
    bootstrap samples of the training data and combines their predictions
    through voting.
    
    Parameters:
    -----------
    base_estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, a simple decision stump is used.
    n_estimators : int, default=10
        The number of base estimators in the ensemble.
    max_samples : float, default=1.0
        The fraction of samples to draw from X to train each base estimator.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_samples_ = []
        
    def _make_estimator(self):
        """Create a new base estimator instance."""
        if self.base_estimator is None:
            return DecisionStump()
        else:
            # Create a copy of the base estimator
            import copy
            return copy.deepcopy(self.base_estimator)
    
    def fit(self, X, y):
        """
        Build a Bagging ensemble of estimators from the training set (X, y).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
            
        Returns:
        --------
        self : object
            Fitted estimator.
        """
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Calculate the number of samples to use for each estimator
        max_samples = int(self.max_samples * n_samples)
        
        self.estimators_ = []
        self.estimator_samples_ = []
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=max_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Fit estimator
            estimator = self._make_estimator()
            estimator.fit(X_sample, y_sample)
            
            self.estimators_.append(estimator)
            self.estimator_samples_.append(indices)
        
        return self
    
    def predict(self, X):
        """
        Predict class for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        X = np.array(X)
        
        # Get predictions from all estimators
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        
        # Majority voting
        y_pred = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            # Get most common prediction
            most_common = Counter(votes).most_common(1)[0][0]
            y_pred.append(most_common)
        
        return np.array(y_pred)
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
            
        Returns:
        --------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class DecisionStump:
    """
    A simple decision tree with only one split (decision stump).
    Used as the default base estimator for bagging.
    """
    
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None
    
    def fit(self, X, y):
        """Fit the decision stump to the training data."""
        X = np.array(X)
        y = np.array(y)
        
        best_gini = float('inf')
        n_samples, n_features = X.shape
        
        # Try all features and all possible thresholds
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate Gini impurity
                gini = self._gini_impurity(y[left_mask], y[right_mask])
                
                if gini < best_gini:
                    best_gini = gini
                    self.feature_index = feature_idx
                    self.threshold = threshold
                    # Most common class in each split
                    self.left_value = Counter(y[left_mask]).most_common(1)[0][0]
                    self.right_value = Counter(y[right_mask]).most_common(1)[0][0]
        
        return self
    
    def predict(self, X):
        """Predict using the decision stump."""
        X = np.array(X)
        predictions = np.zeros(X.shape[0])
        
        left_mask = X[:, self.feature_index] <= self.threshold
        predictions[left_mask] = self.left_value
        predictions[~left_mask] = self.right_value
        
        return predictions
    
    def _gini_impurity(self, left_y, right_y):
        """Calculate Gini impurity for a split."""
        n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right
        
        if n_left == 0 or n_right == 0:
            return float('inf')
        
        # Gini for left split
        left_gini = 1.0
        for cls in np.unique(left_y):
            p = np.sum(left_y == cls) / n_left
            left_gini -= p ** 2
        
        # Gini for right split
        right_gini = 1.0
        for cls in np.unique(right_y):
            p = np.sum(right_y == cls) / n_right
            right_gini -= p ** 2
        
        # Weighted Gini
        gini = (n_left / n_total) * left_gini + (n_right / n_total) * right_gini
        return gini
