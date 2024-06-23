import pandas as pd

def convert_time_column(df):
    # List of possible column names for the time column (in lowercase)
    possible_time_columns = ['date', 'time', 'timestamp', 'datetime', 'date_time', 'date_col', 'time_col']
    
    # Check if any of the possible column names exist in the DataFrame (case-insensitive)
    time_column = None
    for col in df.columns:
        if col.lower() in possible_time_columns:
            time_column = col
            break
    
    if time_column is None:
        raise ValueError("No time column found in the DataFrame.")
    
    # Check if the column is already in datetime format
    if pd.api.types.is_datetime64_any_dtype(df[time_column]):
        return df
    
    # Try to convert the column to datetime
    try:
        df[time_column] = pd.to_datetime(df[time_column])
    except ValueError:
        # If the automatic conversion fails, try other common formats
        date_formats = [
            "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m-%d-%Y", "%m/%d/%Y",
            "%b %d, %Y", "%B %d, %Y", "%b %-d, %Y", "%B %-d, %Y",
            "%b %d, %y", "%B %d, %y", "%b %-d, %y", "%B %-d, %y",
            "%m/%d/%y", "%m/%d/%Y", "%d/%m/%y", "%d/%m/%Y",
            "%Y%m%d", "%y%m%d", "%m%d%Y", "%m%d%y"
        ]
        for fmt in date_formats:
            try:
                df[time_column] = pd.to_datetime(df[time_column], format=fmt, case=False)
                break
            except ValueError:
                pass
    
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        raise ValueError(f"Failed to convert {time_column} to datetime format.")
    
    return df