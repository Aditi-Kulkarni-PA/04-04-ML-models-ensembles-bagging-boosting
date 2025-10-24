"""
Unit tests for Boosting Classifiers
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from ensembles import AdaBoostClassifier, GradientBoostingClassifier
from ensembles.utils import load_sample_data, train_test_split


def test_adaboost_fit():
    """Test that AdaBoostClassifier can fit on training data."""
    X, y = load_sample_data(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    adaboost = AdaBoostClassifier(n_estimators=20, random_state=42)
    adaboost.fit(X_train, y_train)
    
    assert len(adaboost.estimators_) > 0
    assert len(adaboost.estimator_weights_) == len(adaboost.estimators_)
    print("✓ test_adaboost_fit passed")


def test_adaboost_predict():
    """Test that AdaBoostClassifier can make predictions."""
    X, y = load_sample_data(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    adaboost = AdaBoostClassifier(n_estimators=20, random_state=42)
    adaboost.fit(X_train, y_train)
    
    y_pred = adaboost.predict(X_test)
    
    assert len(y_pred) == len(X_test)
    assert set(y_pred).issubset({-1, 0, 1})
    print("✓ test_adaboost_predict passed")


def test_adaboost_score():
    """Test that AdaBoostClassifier can score predictions."""
    X, y = load_sample_data(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    adaboost = AdaBoostClassifier(n_estimators=20, random_state=42)
    adaboost.fit(X_train, y_train)
    
    score = adaboost.score(X_test, y_test)
    
    assert 0 <= score <= 1
    print(f"✓ test_adaboost_score passed (accuracy: {score:.4f})")


def test_gradient_boosting_fit():
    """Test that GradientBoostingClassifier can fit on training data."""
    X, y = load_sample_data(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    gb = GradientBoostingClassifier(n_estimators=20, random_state=42)
    gb.fit(X_train, y_train)
    
    assert len(gb.estimators_) == 20
    print("✓ test_gradient_boosting_fit passed")


def test_gradient_boosting_predict():
    """Test that GradientBoostingClassifier can make predictions."""
    X, y = load_sample_data(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    gb = GradientBoostingClassifier(n_estimators=20, random_state=42)
    gb.fit(X_train, y_train)
    
    y_pred = gb.predict(X_test)
    
    assert len(y_pred) == len(X_test)
    assert set(y_pred).issubset({-1, 1})
    print("✓ test_gradient_boosting_predict passed")


def test_gradient_boosting_score():
    """Test that GradientBoostingClassifier can score predictions."""
    X, y = load_sample_data(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    gb = GradientBoostingClassifier(n_estimators=20, random_state=42)
    gb.fit(X_train, y_train)
    
    score = gb.score(X_test, y_test)
    
    assert 0 <= score <= 1
    print(f"✓ test_gradient_boosting_score passed (accuracy: {score:.4f})")


def test_boosting_performance():
    """Test that boosting classifiers achieve reasonable performance."""
    X, y = load_sample_data(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # AdaBoost
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
    adaboost.fit(X_train, y_train)
    score_adaboost = adaboost.score(X_test, y_test)
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    score_gb = gb.score(X_test, y_test)
    
    # Both should achieve at least 60% accuracy
    assert score_adaboost >= 0.6, f"AdaBoost accuracy {score_adaboost:.4f} < 0.6"
    assert score_gb >= 0.6, f"Gradient Boosting accuracy {score_gb:.4f} < 0.6"
    
    print(f"✓ test_boosting_performance passed")
    print(f"  AdaBoost accuracy: {score_adaboost:.4f}")
    print(f"  Gradient Boosting accuracy: {score_gb:.4f}")


if __name__ == "__main__":
    print("\nRunning Boosting Classifier Tests...")
    print("=" * 60)
    
    test_adaboost_fit()
    test_adaboost_predict()
    test_adaboost_score()
    test_gradient_boosting_fit()
    test_gradient_boosting_predict()
    test_gradient_boosting_score()
    test_boosting_performance()
    
    print("=" * 60)
    print("All Boosting tests passed!\n")
