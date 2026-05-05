"""
Missing Detection Package
A custom library for detecting missing values in pandas DataFrames with flexible patterns.
"""

from .core import (
    MISSING_PATTERNS,
    normalize_df,
    detect_missing,
    clean_missing,
    missing_report,
    filter_missing
)

__version__ = "0.1.0"