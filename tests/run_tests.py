"""
Test runner for all ensemble tests
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import test modules
from test_bagging import (
    test_bagging_fit,
    test_bagging_predict,
    test_bagging_score,
    test_bagging_performance
)

from test_boosting import (
    test_adaboost_fit,
    test_adaboost_predict,
    test_adaboost_score,
    test_gradient_boosting_fit,
    test_gradient_boosting_predict,
    test_gradient_boosting_score,
    test_boosting_performance
)


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING ALL ENSEMBLE LEARNING TESTS")
    print("=" * 70)
    
    # Bagging tests
    print("\n1. Bagging Classifier Tests")
    print("-" * 70)
    test_bagging_fit()
    test_bagging_predict()
    test_bagging_score()
    test_bagging_performance()
    
    # Boosting tests
    print("\n2. Boosting Classifiers Tests")
    print("-" * 70)
    test_adaboost_fit()
    test_adaboost_predict()
    test_adaboost_score()
    test_gradient_boosting_fit()
    test_gradient_boosting_predict()
    test_gradient_boosting_score()
    test_boosting_performance()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! ✓")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
