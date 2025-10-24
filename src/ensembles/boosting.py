"""
Boosting Algorithms: AdaBoost and Gradient Boosting
"""

import numpy as np
from collections import Counter


class AdaBoostClassifier:
    """
    AdaBoost (Adaptive Boosting) classifier.
    
    AdaBoost fits a sequence of weak learners on repeatedly modified versions
    of the data. The predictions are combined through a weighted majority vote.
    
    Parameters:
    -----------
    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
    learning_rate : float, default=1.0
        Weight applied to each classifier at each boosting iteration.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        
    def fit(self, X, y):
        """
        Build a boosted classifier from the training set (X, y).
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
            
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
        
        # Initialize weights uniformly
        sample_weights = np.ones(n_samples) / n_samples
        
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        
        for i in range(self.n_estimators):
            # Fit a weak learner
            estimator = DecisionStump()
            estimator.fit(X, y, sample_weight=sample_weights)
            
            # Get predictions
            y_pred = estimator.predict(X)
            
            # Calculate weighted error
            incorrect = (y_pred != y)
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
            
            # Stop if error is too high or too low
            if error >= 0.5:
                if len(self.estimators_) == 0:
                    self.estimators_.append(estimator)
                    self.estimator_weights_.append(1.0)
                    self.estimator_errors_.append(error)
                break
            
            if error <= 0:
                self.estimators_.append(estimator)
                self.estimator_weights_.append(1.0)
                self.estimator_errors_.append(error)
                break
            
            # Calculate estimator weight
            estimator_weight = self.learning_rate * 0.5 * np.log((1 - error) / error)
            
            # Update sample weights
            sample_weights = sample_weights * np.exp(estimator_weight * incorrect * 
                                                      ((sample_weights > 0) | (estimator_weight < 0)))
            
            # Normalize weights
            sample_weights = sample_weights / np.sum(sample_weights)
            
            self.estimators_.append(estimator)
            self.estimator_weights_.append(estimator_weight)
            self.estimator_errors_.append(error)
        
        return self
    
    def predict(self, X):
        """
        Predict classes for X.
        
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
        n_samples = X.shape[0]
        
        # Get weighted predictions from all estimators
        predictions = np.zeros(n_samples)
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            predictions += weight * estimator.predict(X)
        
        # Return sign of weighted sum (assuming binary classification with -1, 1)
        return np.sign(predictions)
    
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


class GradientBoostingClassifier:
    """
    Gradient Boosting for classification.
    
    Gradient Boosting builds an additive model in a forward stage-wise fashion;
    it allows for the optimization of arbitrary differentiable loss functions.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        The number of boosting stages to perform.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree.
    max_depth : int, default=3
        Maximum depth of the individual regression estimators.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []
        self.init_prediction_ = None
        
    def fit(self, X, y):
        """
        Fit the gradient boosting model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
            
        Returns:
        --------
        self : object
            Fitted estimator.
        """
        X = np.array(X)
        y = np.array(y)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Initialize with log-odds for binary classification
        positive_count = np.sum(y == 1)
        negative_count = np.sum(y == -1)
        self.init_prediction_ = np.log(positive_count / negative_count) if negative_count > 0 else 0.0
        
        # Initialize predictions
        predictions = np.full(len(y), self.init_prediction_)
        
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # Calculate residuals (negative gradient for log loss)
            # For binary classification: residual = y - sigmoid(prediction)
            probabilities = self._sigmoid(predictions)
            residuals = (y - 2 * probabilities + 1) / 2  # Convert to [0, 1] space
            
            # Fit a tree to residuals
            tree = SimpleRegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Update predictions
            update = tree.predict(X)
            predictions += self.learning_rate * update
            
            self.estimators_.append(tree)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        proba : array-like of shape (n_samples,)
            The class probabilities.
        """
        X = np.array(X)
        
        # Start with initial prediction
        predictions = np.full(X.shape[0], self.init_prediction_)
        
        # Add contributions from all trees
        for tree in self.estimators_:
            predictions += self.learning_rate * tree.predict(X)
        
        # Convert to probabilities
        return self._sigmoid(predictions)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        y : array-like of shape (n_samples,)
            The predicted class labels.
        """
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, -1)
    
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
    
    def _sigmoid(self, x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class SimpleRegressionTree:
    """
    A simple regression tree for gradient boosting.
    """
    
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
        
    def fit(self, X, y, sample_weight=None):
        """Fit the regression tree."""
        X = np.array(X)
        y = np.array(y)
        
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        
        self.tree = self._build_tree(X, y, sample_weight, depth=0)
        return self
    
    def _build_tree(self, X, y, sample_weight, depth):
        """Recursively build the tree."""
        # If max depth reached or too few samples, make leaf node
        if depth >= self.max_depth or len(y) < 2:
            return {'value': np.average(y, weights=sample_weight)}
        
        best_gain = 0
        best_split = None
        current_mse = self._weighted_mse(y, sample_weight)
        
        n_samples, n_features = X.shape
        
        # Try all features and thresholds
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate weighted MSE for split
                left_mse = self._weighted_mse(y[left_mask], sample_weight[left_mask])
                right_mse = self._weighted_mse(y[right_mask], sample_weight[right_mask])
                
                # Weighted average of MSEs
                total_weight = np.sum(sample_weight)
                weighted_mse = (np.sum(sample_weight[left_mask]) * left_mse + 
                               np.sum(sample_weight[right_mask]) * right_mse) / total_weight
                
                gain = current_mse - weighted_mse
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }
        
        # If no good split found, make leaf
        if best_split is None:
            return {'value': np.average(y, weights=sample_weight)}
        
        # Build left and right subtrees
        left_tree = self._build_tree(
            X[best_split['left_mask']], 
            y[best_split['left_mask']], 
            sample_weight[best_split['left_mask']],
            depth + 1
        )
        right_tree = self._build_tree(
            X[best_split['right_mask']], 
            y[best_split['right_mask']], 
            sample_weight[best_split['right_mask']],
            depth + 1
        )
        
        return {
            'feature_idx': best_split['feature_idx'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }
    
    def predict(self, X):
        """Predict using the regression tree."""
        X = np.array(X)
        return np.array([self._predict_single(x, self.tree) for x in X])
    
    def _predict_single(self, x, node):
        """Predict for a single sample."""
        if 'value' in node:
            return node['value']
        
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])
    
    def _weighted_mse(self, y, weights):
        """Calculate weighted mean squared error."""
        if len(y) == 0:
            return 0
        weighted_mean = np.average(y, weights=weights)
        return np.average((y - weighted_mean) ** 2, weights=weights)


class DecisionStump:
    """
    A simple decision stump for AdaBoost.
    """
    
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None
    
    def fit(self, X, y, sample_weight=None):
        """Fit the decision stump."""
        X = np.array(X)
        y = np.array(y)
        
        if sample_weight is None:
            sample_weight = np.ones(len(y)) / len(y)
        else:
            sample_weight = np.array(sample_weight)
        
        best_error = float('inf')
        n_samples, n_features = X.shape
        
        # Try all features and thresholds
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                # Try both polarities
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    predictions[X[:, feature_idx] <= threshold] = -1
                    if polarity == -1:
                        predictions = -predictions
                    
                    # Calculate weighted error
                    error = np.sum(sample_weight * (predictions != y))
                    
                    if error < best_error:
                        best_error = error
                        self.feature_index = feature_idx
                        self.threshold = threshold
                        self.polarity = polarity
        
        return self
    
    def predict(self, X):
        """Predict using the decision stump."""
        X = np.array(X)
        predictions = np.ones(X.shape[0])
        predictions[X[:, self.feature_index] <= self.threshold] = -1
        if hasattr(self, 'polarity') and self.polarity == -1:
            predictions = -predictions
        return predictions
