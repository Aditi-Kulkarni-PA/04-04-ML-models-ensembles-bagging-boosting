"""
Comparison script for ensemble methods.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from ensembles import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from ensembles.utils import load_sample_data, train_test_split, compute_metrics


def compare_methods():
    """Compare different ensemble methods on various datasets."""
    
    print("\n" + "=" * 80)
    print("Ensemble Methods Comparison")
    print("=" * 80)
    
    # Test on different dataset sizes
    dataset_configs = [
        {"n_samples": 200, "n_features": 5, "name": "Small Dataset"},
        {"n_samples": 500, "n_features": 10, "name": "Medium Dataset"},
        {"n_samples": 1000, "n_features": 15, "name": "Large Dataset"},
    ]
    
    results = []
    
    for config in dataset_configs:
        print(f"\n{config['name']}: {config['n_samples']} samples, {config['n_features']} features")
        print("-" * 80)
        
        # Load data
        X, y = load_sample_data(
            n_samples=config['n_samples'],
            n_features=config['n_features'],
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Bagging
        bagging = BaggingClassifier(n_estimators=15, max_samples=0.8, random_state=42)
        bagging.fit(X_train, y_train)
        acc_bagging = bagging.score(X_test, y_test)
        print(f"  Bagging:           Accuracy = {acc_bagging:.4f}")
        
        # AdaBoost
        adaboost = AdaBoostClassifier(n_estimators=30, learning_rate=1.0, random_state=42)
        adaboost.fit(X_train, y_train)
        acc_adaboost = adaboost.score(X_test, y_test)
        print(f"  AdaBoost:          Accuracy = {acc_adaboost:.4f}")
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
        gb.fit(X_train, y_train)
        acc_gb = gb.score(X_test, y_test)
        print(f"  Gradient Boosting: Accuracy = {acc_gb:.4f}")
        
        results.append({
            'dataset': config['name'],
            'bagging': acc_bagging,
            'adaboost': acc_adaboost,
            'gradient_boosting': acc_gb
        })
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary of Results")
    print("=" * 80)
    print(f"{'Dataset':<20} {'Bagging':<20} {'AdaBoost':<20} {'Gradient Boosting':<20}")
    print("-" * 80)
    for result in results:
        print(f"{result['dataset']:<20} {result['bagging']:<20.4f} "
              f"{result['adaboost']:<20.4f} {result['gradient_boosting']:<20.4f}")
    print("=" * 80)
    
    # Calculate average performance
    avg_bagging = np.mean([r['bagging'] for r in results])
    avg_adaboost = np.mean([r['adaboost'] for r in results])
    avg_gb = np.mean([r['gradient_boosting'] for r in results])
    
    print(f"\nAverage Accuracy:")
    print(f"  Bagging:           {avg_bagging:.4f}")
    print(f"  AdaBoost:          {avg_adaboost:.4f}")
    print(f"  Gradient Boosting: {avg_gb:.4f}")
    
    print("\n✓ Comparison completed successfully!")


if __name__ == "__main__":
    compare_methods()
