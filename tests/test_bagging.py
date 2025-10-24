"""
Unit tests for Bagging Classifier
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from ensembles import BaggingClassifier
from ensembles.utils import load_sample_data, train_test_split


def test_bagging_fit():
    """Test that BaggingClassifier can fit on training data."""
    X, y = load_sample_data(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    bagging = BaggingClassifier(n_estimators=10, random_state=42)
    bagging.fit(X_train, y_train)
    
    assert len(bagging.estimators_) == 10
    print("✓ test_bagging_fit passed")


def test_bagging_predict():
    """Test that BaggingClassifier can make predictions."""
    X, y = load_sample_data(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    bagging = BaggingClassifier(n_estimators=10, random_state=42)
    bagging.fit(X_train, y_train)
    
    y_pred = bagging.predict(X_test)
    
    assert len(y_pred) == len(X_test)
    assert set(y_pred).issubset({-1, 1})
    print("✓ test_bagging_predict passed")


def test_bagging_score():
    """Test that BaggingClassifier can score predictions."""
    X, y = load_sample_data(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    bagging = BaggingClassifier(n_estimators=10, random_state=42)
    bagging.fit(X_train, y_train)
    
    score = bagging.score(X_test, y_test)
    
    assert 0 <= score <= 1
    print(f"✓ test_bagging_score passed (accuracy: {score:.4f})")


def test_bagging_performance():
    """Test that BaggingClassifier achieves reasonable performance."""
    X, y = load_sample_data(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    bagging = BaggingClassifier(n_estimators=20, random_state=42)
    bagging.fit(X_train, y_train)
    
    score = bagging.score(X_test, y_test)
    
    # Should achieve at least 60% accuracy on this dataset
    assert score >= 0.6, f"Expected accuracy >= 0.6, got {score:.4f}"
    print(f"✓ test_bagging_performance passed (accuracy: {score:.4f})")


if __name__ == "__main__":
    print("\nRunning Bagging Classifier Tests...")
    print("=" * 60)
    
    test_bagging_fit()
    test_bagging_predict()
    test_bagging_score()
    test_bagging_performance()
    
    print("=" * 60)
    print("All Bagging tests passed!\n")
