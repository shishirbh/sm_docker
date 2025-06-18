"""
Unified SageMaker Framework
A flexible framework for custom training and serving on Amazon SageMaker.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from . import training
from . import serving

__all__ = ['training', 'serving']