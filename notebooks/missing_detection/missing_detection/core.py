import pandas as pd
import numpy as np

MISSING_PATTERNS = {
    "", " ", "  ",        # empty / whitespace
    "na", "n/a", "nan",   # text nulls
    "null", "none",
    "-", "--",
    "/0", "0/0",
    '""', "''"
}

def normalize_df(df):
    """
    Normalize DataFrame for missing detection by converting to string,
    stripping whitespace, and lowercasing.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: Normalized DataFrame
    """
    df = df.copy()

    # Convert all columns to string for uniform cleaning
    df = df.astype(str)

    # Strip whitespace
    df = df.apply(lambda col: col.str.strip())

    # Lowercase for consistency
    df = df.apply(lambda col: col.str.lower())

    return df

def detect_missing(df):
    """
    Detect missing values using multiple criteria:
    - Standard pandas NaN/None
    - Custom patterns (empty strings, 'na', etc.)
    - Regex patterns (whitespace-only, quotes-only)

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: Boolean mask where True indicates missing value
    """
    df_norm = normalize_df(df)

    # Standard pandas missing
    mask_standard = df.isna()

    # Pattern-based missing
    mask_patterns = df_norm.isin(MISSING_PATTERNS)

    # Regex-based detection (advanced cases)
    mask_regex = df_norm.apply(
        lambda col: col.str.match(r'^\s*$|^["\']+$', na=False)
    )

    # Combine all masks
    mask_all = mask_standard | mask_patterns | mask_regex

    return mask_all

def clean_missing(df):
    """
    Replace detected missing values with np.nan.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: DataFrame with missing values as np.nan
    """
    mask = detect_missing(df)
    df_clean = df.mask(mask, np.nan)
    return df_clean

def missing_report(df):
    """
    Generate a report of missing values per column.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: Report with missing count and percentage per column
    """
    mask = detect_missing(df)

    report = pd.DataFrame({
        "missing_count": mask.sum(),
        "missing_percent": (mask.sum() / len(df)) * 100
    })

    report = report.sort_values(by="missing_percent", ascending=False)
    print(report)
    return report

def filter_missing(df, columns=None, how="any"):
    """
    Filter rows that contain missing values.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list, optional): Columns to check for missing values. If None, checks all.
    how (str): 'any' or 'all' - if 'any', filter rows with any missing in columns;
               if 'all', filter rows with all missing in columns.

    Returns:
    pd.DataFrame: Filtered DataFrame with rows containing missing values
    """
    mask = detect_missing(df)
    if columns is None:
        subset = mask
    else:
        subset = mask[columns]
    if how == "any":
        return df[subset.any(axis=1)]
    elif how == "all":
        return df[subset.all(axis=1)]
    else:
        raise ValueError("how must be 'any' or 'all'")