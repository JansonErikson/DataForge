import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from test_utils import convert_time_column

class DataFrameAnalyzer:
    def __init__(self, df):
        self.df = convert_time_column(df)  # Convert time column to datetime format
        
    def analyze(self):
        print("Data types:")
        print(self.df.dtypes)
        print()
        
        print("Number of rows:")
        print(len(self.df))
        print()
        print("Number of rows of each column:")
        print(self.df.count())
        print()
        
        print("First and last value of each column (if datetime):")
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                print(f"{col}: {self.df[col].min()} to {self.df[col].max()}")
        print()
        
        print("Missing values per column:")
        print(self.df.isnull().sum())
        print()
        
        print("Outliers per column (values outside 1.5 times the interquartile range):")
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                print(f"{col}: {len(outliers)} outliers")
        print()
        
        print("Scale of values per column:")
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                print(f"{col}: Min={self.df[col].min()}, Max={self.df[col].max()}")
        print()
        
        print("Best time window with available values for all columns:")
        complete_data = self.df.dropna()
        if not complete_data.empty:
            start_date = complete_data.iloc[0]
            end_date = complete_data.iloc[-1]
            print(f"From {start_date.name} ({start_date.iloc[0]}) to {end_date.name} ({end_date.iloc[0]})")
        else:
            print("No complete data found.")
        print()
        

    def fill_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'constant' and fill_value is None:
            raise ValueError("fill_value must be provided when strategy is 'constant'.")
        
        if strategy == 'linear':
            self.df.interpolate(method='linear', inplace=True)
        else:
            imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
            numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
            self.df[numeric_columns] = imputer.fit_transform(self.df[numeric_columns])
        return self.df
    
    def auto_scale_data(self):
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numeric_columns:
            if (self.df[col] >= 0).all() and (self.df[col] <= 1).all():
                # All values are between 0 and 1, no scaling needed
                continue
            elif (self.df[col] > 1).any():
                # Some values are greater than 1, use MinMaxScaler
                scaler = MinMaxScaler()
            else:
                # Values are less than 1 and greater than -1, use StandardScaler
                scaler = StandardScaler()
            
            self.df[col] = scaler.fit_transform(self.df[[col]])
        
        return self.df
    
    def preprocess_data(self, fill_strategy='mean', fill_value=None, scale_data=True):
        self.fill_missing_values(strategy=fill_strategy, fill_value=fill_value)
        if scale_data:
            self.auto_scale_data()
        return self.df
    
    def display_correlation_matrix(self, method='pearson', figsize=(10, 8), annot=True, cmap='coolwarm'):
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = self.df[numeric_columns].corr(method=method)
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=annot, fmt='.2f')
        
        plt.title(f'Correlation Matrix ({method.capitalize()} Correlation)')
        plt.show()
    
