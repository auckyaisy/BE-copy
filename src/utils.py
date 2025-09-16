"""
Utility functions for the well analysis pipeline.
"""
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple
import json
import yaml
import pandas as pd
import numpy as np
from scipy import stats


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to the log file. If None, logs to console only.
        level: Logging level (default: logging.INFO)
    """
    log_handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )


def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.
    
    Args:
        config_file: Path to the configuration file.
        
    Returns:
        Dict containing the configuration.
    """
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        if config_file.suffix.lower() == '.json':
            return json.load(f)
        elif config_file.suffix.lower() in ('.yaml', '.yml'):
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")


def save_config(config: Dict[str, Any], config_file: Union[str, Path]) -> None:
    """
    Save configuration to a JSON or YAML file.
    
    Args:
        config: Configuration dictionary to save.
        config_file: Path to save the configuration file.
    """
    config_file = Path(config_file)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        if config_file.suffix.lower() == '.json':
            json.dump(config, f, indent=4)
        elif config_file.suffix.lower() in ('.yaml', '.yml'):
            yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory.
        
    Returns:
        Path object of the directory.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by converting to lowercase and replacing spaces with underscores.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        DataFrame with cleaned column names.
    """
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df


def resample_time_series(df: pd.DataFrame, 
                        time_col: str = 'reading_time', 
                        freq: str = '1H',
                        agg_func: str = 'mean') -> pd.DataFrame:
    """
    Resample a time series DataFrame to a specified frequency.
    
    Args:
        df: Input DataFrame with a datetime column.
        time_col: Name of the datetime column.
        freq: Resampling frequency (e.g., '1H' for hourly, '1D' for daily).
        agg_func: Aggregation function to use ('mean', 'sum', 'first', 'last', etc.).
        
    Returns:
        Resampled DataFrame.
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame.")
    
    # Convert to datetime if not already
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Set the time column as index
    df = df.set_index(time_col)
    
    # Resample the data
    if agg_func == 'mean':
        df_resampled = df.resample(freq).mean()
    elif agg_func == 'sum':
        df_resampled = df.resample(freq).sum()
    elif agg_func == 'first':
        df_resampled = df.resample(freq).first()
    elif agg_func == 'last':
        df_resampled = df.resample(freq).last()
    else:
        raise ValueError(f"Unsupported aggregation function: {agg_func}")
    
    # Reset index to make time a column again
    df_resampled = df_resampled.reset_index()
    
    return df_resampled


def calculate_statistics(df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate basic statistics for numerical columns in a DataFrame.
    
    Args:
        df: Input DataFrame.
        group_col: Column to group by before calculating statistics.
        
    Returns:
        DataFrame containing the calculated statistics.
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if group_col:
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in DataFrame.")
        
        # Group by the specified column and calculate statistics
        stats = df.groupby(group_col)[numeric_cols].agg(['count', 'mean', 'std', 'min', 'max'])
        # Flatten multi-index columns
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    else:
        # Calculate statistics for the entire DataFrame
        stats = pd.DataFrame()
        for col in numeric_cols:
            stats[f'{col}_count'] = [df[col].count()]
            stats[f'{col}_mean'] = [df[col].mean()]
            stats[f'{col}_std'] = [df[col].std()]
            stats[f'{col}_min'] = [df[col].min()]
            stats[f'{col}_max'] = [df[col].max()]
    
    return stats


def save_dataframe(df: pd.DataFrame, file_path: Union[str, Path], index: bool = False) -> None:
    """
    Save a DataFrame to a file, automatically detecting the format from the file extension.
    
    Args:
        df: DataFrame to save.
        file_path: Path where to save the file.
        index: Whether to write row names (index).
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_path.suffix.lower() == '.csv':
        df.to_csv(file_path, index=index)
    elif file_path.suffix.lower() == '.xlsx':
        df.to_excel(file_path, index=index)
    elif file_path.suffix.lower() == '.parquet':
        df.to_parquet(file_path, index=index)
    elif file_path.suffix.lower() == '.feather':
        df.to_feather(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logging.info(f"Saved DataFrame to {file_path}")


def load_dataframe(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load a DataFrame from a file, automatically detecting the format from the file extension.
    
    Args:
        file_path: Path to the file to load.
        **kwargs: Additional arguments to pass to the pandas read function.
        
    Returns:
        Loaded DataFrame.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() == '.csv':
        return pd.read_csv(file_path, **kwargs)
    elif file_path.suffix.lower() == '.xlsx':
        return pd.read_excel(file_path, **kwargs)
    elif file_path.suffix.lower() == '.parquet':
        return pd.read_parquet(file_path, **kwargs)
    elif file_path.suffix.lower() == '.feather':
        return pd.read_feather(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def calculate_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the slope of a line of best fit for given x and y values.
    
    Args:
        x: Array of x-values
        y: Array of y-values (same length as x)
        
    Returns:
        Slope of the line of best fit
    """
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    return np.polyfit(x, y, 1)[0]


def calculate_well_slopes(
    df: pd.DataFrame,
    value_column: str,
    time_column: str = 'Reading Time',
    window: int = 3
) -> pd.DataFrame:
    """
    Calculate rolling slopes for well data.
    
    Args:
        df: DataFrame containing the well data
        value_column: Name of the column containing values to calculate slopes for
        time_column: Name of the column containing timestamps
        window: Number of points to include in the rolling window for slope calculation
        
    Returns:
        DataFrame with an additional 'Slope' column
    """
    if df.empty or value_column not in df.columns:
        return df
    
    # Make a copy to avoid SettingWithCopyWarning
    result = df.copy()
    
    # Ensure the time column is datetime
    if time_column in result.columns:
        result[time_column] = pd.to_datetime(result[time_column])
        result = result.sort_values(time_column)
    
    # Initialize slope column
    result['Slope'] = 0.0
    
    # Calculate slopes using a rolling window
    for i in range(window - 1, len(result)):
        window_data = result.iloc[i - window + 1:i + 1]
        if len(window_data) < 2:
            continue
            
        if time_column in result.columns:
            # Convert timestamps to numeric values for slope calculation
            x = (window_data[time_column] - window_data[time_column].iloc[0]).dt.total_seconds().values
        else:
            x = np.arange(len(window_data))
            
        y = window_data[value_column].values
        result.iloc[i, result.columns.get_loc('Slope')] = calculate_slope(x, y)
    
    return result
