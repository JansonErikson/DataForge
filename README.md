# DataFrame Analyzer

DataFrame Analyzer is a Python package that provides a set of tools for analyzing and preprocessing DataFrames. It offers functionality to perform comprehensive analysis of DataFrame columns, fill missing values, scale data, and visualize correlation matrices.

## Installation

You can install DataFrame Analyzer using pip:

pip install dataframe-analyzer

## Usage

Here's a basic usage example:

```python
from dataframe_analyzer import DataFrameAnalyzer

# Create a DataFrameAnalyzer object
analyzer = DataFrameAnalyzer(your_dataframe)

# Perform comprehensive analysis
analyzer.analyze()

# Fill missing values
analyzer.fill_missing_values(strategy='mean')

# Scale the data
analyzer.auto_scale_data()

# Display the correlation matrix
analyzer.display_correlation_matrix()