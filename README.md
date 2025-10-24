# Ensemble Learning: Bagging and Boosting

A comprehensive implementation of ensemble learning methods including Bagging, AdaBoost, and Gradient Boosting algorithms from scratch using Python and NumPy.

## Overview

This project implements three popular ensemble learning techniques:

- **Bagging (Bootstrap Aggregating)**: Trains multiple models on different bootstrap samples and combines predictions through voting
- **AdaBoost (Adaptive Boosting)**: Sequentially trains weak learners, focusing on previously misclassified samples
- **Gradient Boosting**: Builds an additive model by optimizing differentiable loss functions

## Features

- ✅ Pure Python/NumPy implementations
- ✅ Scikit-learn compatible API
- ✅ Comprehensive test suite
- ✅ Example scripts and demos
- ✅ Well-documented code

## Project Structure

```
ensembles-bagging-boosting/
├── src/
│   └── ensembles/
│       ├── __init__.py          # Package initialization
│       ├── bagging.py           # Bagging classifier implementation
│       ├── boosting.py          # AdaBoost and Gradient Boosting
│       └── utils.py             # Utility functions
├── tests/
│   ├── test_bagging.py          # Bagging tests
│   ├── test_boosting.py         # Boosting tests
│   └── run_tests.py             # Test runner
├── examples/
│   ├── demo.py                  # Basic usage demo
│   └── comparison.py            # Performance comparison
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Aditi-Kulkarni-PA/ensembles-bagging-boosting.git
cd ensembles-bagging-boosting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from ensembles import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from ensembles.utils import load_sample_data, train_test_split

# Load data
X, y = load_sample_data(n_samples=300, n_features=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Bagging
bagging = BaggingClassifier(n_estimators=20, random_state=42)
bagging.fit(X_train, y_train)
print(f"Bagging Accuracy: {bagging.score(X_test, y_test):.4f}")

# AdaBoost
adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
adaboost.fit(X_train, y_train)
print(f"AdaBoost Accuracy: {adaboost.score(X_test, y_test):.4f}")

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)
print(f"Gradient Boosting Accuracy: {gb.score(X_test, y_test):.4f}")
```

### Running Examples

Run the demo script:
```bash
python examples/demo.py
```

Run the comparison script:
```bash
python examples/comparison.py
```

## Running Tests

Run all tests:
```bash
python tests/run_tests.py
```

Run specific test files:
```bash
python tests/test_bagging.py
python tests/test_boosting.py
```

## Algorithm Details

### Bagging (Bootstrap Aggregating)

Bagging reduces variance by:
1. Creating multiple bootstrap samples from the training data
2. Training a model on each bootstrap sample
3. Combining predictions through majority voting

**Parameters:**
- `n_estimators`: Number of base estimators (default: 10)
- `max_samples`: Fraction of samples to draw for each estimator (default: 1.0)
- `random_state`: Random seed for reproducibility

### AdaBoost (Adaptive Boosting)

AdaBoost reduces bias by:
1. Training weak learners sequentially
2. Adjusting sample weights to focus on misclassified examples
3. Combining predictions with weighted voting

**Parameters:**
- `n_estimators`: Maximum number of estimators (default: 50)
- `learning_rate`: Weight applied to each classifier (default: 1.0)
- `random_state`: Random seed for reproducibility

### Gradient Boosting

Gradient Boosting optimizes loss functions by:
1. Starting with an initial prediction
2. Fitting trees to the negative gradient (residuals)
3. Adding trees sequentially with a learning rate

**Parameters:**
- `n_estimators`: Number of boosting stages (default: 100)
- `learning_rate`: Shrinks the contribution of each tree (default: 0.1)
- `max_depth`: Maximum depth of trees (default: 3)
- `random_state`: Random seed for reproducibility

## Performance

Typical accuracy on sample datasets:
- Bagging: 85-92%
- AdaBoost: 88-95%
- Gradient Boosting: 90-96%

## Requirements

- Python 3.7+
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0 (for data generation and utilities only)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## References

- Breiman, L. (1996). "Bagging predictors". Machine Learning.
- Freund, Y., & Schapire, R. E. (1997). "A decision-theoretic generalization of on-line learning and an application to boosting".
- Friedman, J. H. (2001). "Greedy function approximation: a gradient boosting machine".
