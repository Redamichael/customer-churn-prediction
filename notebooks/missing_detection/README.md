# Missing Detection Package

A custom Python library for detecting missing values in pandas DataFrames with flexible patterns beyond standard NaN detection.

## Features

- Detects missing values using multiple criteria:
  - Standard pandas NaN/None values
  - Custom patterns (empty strings, 'na', 'n/a', 'null', etc.)
  - Regex patterns (whitespace-only strings, quote-only strings)
- Generates missing value reports
- Cleans data by replacing detected missing values with NaN
- Filters rows containing missing values

## Installation

### From source
```bash
cd missing_detection
pip install -e .
```

### Or install dependencies manually
```bash
pip install pandas numpy
```

Then copy the `missing_detection` folder to your Python path.

## Usage

```python
import pandas as pd
from missing_detection import detect_missing, missing_report, clean_missing, filter_missing

# Load your data
df = pd.read_csv('your_data.csv')

# Generate missing report
report = missing_report(df)
print(report)

# Clean missing values
df_clean = clean_missing(df)

# Filter rows with missing values
missing_rows = filter_missing(df)
```

## API Reference

### Functions

- `detect_missing(df)`: Returns boolean mask of missing values
- `missing_report(df)`: Returns DataFrame with missing counts and percentages
- `clean_missing(df)`: Replaces detected missing values with np.nan
- `filter_missing(df, columns=None, how='any')`: Filters rows with missing values

### Configuration

Modify `MISSING_PATTERNS` in `core.py` to customize what patterns are considered missing.

## License

MIT License