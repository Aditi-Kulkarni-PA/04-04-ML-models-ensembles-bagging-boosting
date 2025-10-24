"""
Ensemble Learning Methods: Bagging and Boosting
"""

from .bagging import BaggingClassifier
from .boosting import AdaBoostClassifier, GradientBoostingClassifier

__all__ = [
    'BaggingClassifier',
    'AdaBoostClassifier',
    'GradientBoostingClassifier'
]

__version__ = '1.0.0'
