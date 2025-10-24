"""
Example script demonstrating the usage of ensemble methods.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ensembles import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from ensembles.utils import load_sample_data, train_test_split, compute_metrics, print_metrics


def main():
    """Main function to demonstrate ensemble methods."""
    
    print("=" * 60)
    print("Ensemble Learning Methods: Bagging and Boosting Demo")
    print("=" * 60)
    
    # Load sample data
    print("\n1. Loading sample data...")
    X, y = load_sample_data(n_samples=300, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples:  {len(X_test)}")
    print(f"   Features:         {X.shape[1]}")
    
    # Bagging Classifier
    print("\n2. Training Bagging Classifier...")
    bagging = BaggingClassifier(n_estimators=20, max_samples=0.8, random_state=42)
    bagging.fit(X_train, y_train)
    
    y_pred_bagging = bagging.predict(X_test)
    metrics_bagging = compute_metrics(y_test, y_pred_bagging)
    print_metrics(metrics_bagging, "Bagging Classifier")
    
    # AdaBoost Classifier
    print("\n3. Training AdaBoost Classifier...")
    adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
    adaboost.fit(X_train, y_train)
    
    y_pred_adaboost = adaboost.predict(X_test)
    metrics_adaboost = compute_metrics(y_test, y_pred_adaboost)
    print_metrics(metrics_adaboost, "AdaBoost Classifier")
    
    # Gradient Boosting Classifier
    print("\n4. Training Gradient Boosting Classifier...")
    gradient_boosting = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3,
        random_state=42
    )
    gradient_boosting.fit(X_train, y_train)
    
    y_pred_gb = gradient_boosting.predict(X_test)
    metrics_gb = compute_metrics(y_test, y_pred_gb)
    print_metrics(metrics_gb, "Gradient Boosting Classifier")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("-" * 60)
    print(f"{'Method':<30} {'Accuracy':<15} {'F1-Score':<15}")
    print("-" * 60)
    print(f"{'Bagging':<30} {metrics_bagging['accuracy']:<15.4f} {metrics_bagging['f1_score']:<15.4f}")
    print(f"{'AdaBoost':<30} {metrics_adaboost['accuracy']:<15.4f} {metrics_adaboost['f1_score']:<15.4f}")
    print(f"{'Gradient Boosting':<30} {metrics_gb['accuracy']:<15.4f} {metrics_gb['f1_score']:<15.4f}")
    print("=" * 60)
    
    print("\n✓ Demo completed successfully!")


if __name__ == "__main__":
    main()
