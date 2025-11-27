import logging
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
from typing import Dict, Tuple, Optional, Union, List
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import textwrap
from datetime import timedelta

from config.config import (
    MODEL_PATHS, FEATURE_COLUMNS, TARGET_COLUMNS,
    INPUT_DIR, OUTPUT_DIR, DEFAULT_PARAMS
)
from .utils import calculate_well_slopes

logger = logging.getLogger(__name__)

class WellAnalysisPipeline:
    """
    A pipeline for processing well data and making predictions using trained models.
    """
    
    def __init__(self, well_name: str, output_dir: Optional[Path] = None):
        """
        Initialize the pipeline with a specific well name.
        
        Args:
            well_name: Name of the well (used for input/output file naming)
            output_dir: Optional custom directory to save all outputs
        """
        self.well_name = well_name
        self.models = {}
        self.data = None
        self.predictions = {}
        self.df_wc: Optional[pd.DataFrame] = None
        # Resolve and create output directory
        self.output_dir: Path = Path(output_dir) if output_dir is not None else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Optional paths tracked for flexible Watercut loading
        self.prod_data_path: Optional[Path] = None
        self.input_file_path: Optional[Path] = None
    
    def load_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load well data from a CSV file.
        
        Args:
            file_path: Path to the input CSV file. If None, looks in the default input directory.
            
        Returns:
            pd.DataFrame: Loaded data
        """
        if file_path is None:
            file_path = INPUT_DIR / f"{self.well_name}.csv"
        else:
            # Convert to Path if string
            file_path = Path(file_path)
        
        logger.info(f"Loading data from {file_path}")
        self.data = pd.read_csv(file_path)
        try:
            self.input_file_path = Path(file_path).resolve()
        except Exception:
            self.input_file_path = Path(file_path)
        return self.data
    
    def preprocess_data(self, target_model: str = 'discharge_pressure') -> pd.DataFrame:
        """
        Preprocess the data for a specific model.
        
        Args:
            target_model: The target model for preprocessing
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info(f"Preprocessing data for {target_model}")
        
        # Make a copy of the data to avoid modifying the original
        df = self.data.copy()
        
        # Convert timestamp to datetime if it exists
        if 'Reading Time' in df.columns:
            df['Reading Time'] = pd.to_datetime(df['Reading Time'], format='mixed', dayfirst=False)
            df = df.sort_values('Reading Time')
        
        # Handle missing values
        df = self._handle_missing_values(df, target_model)
        
        # Feature engineering
        df = self._feature_engineering(df, target_model)
        
        return df
    
    def _handle_missing_columns(self, df: pd.DataFrame, target_model: str) -> pd.DataFrame:
        """Ensure all required columns are present with default values."""
        # Define default values for each column type
        default_values = {
            'Average Amps (A) (Raw)': 50.0,
            'Drive Frequency (Hz) (Raw)': 50.0,
            'Intake Pressure (psi) (Raw)': 300.0,
            'Intake Temperature (F) (Raw)': 150.0,
            'Motor Temperature (F) (Raw)': 180.0,
            'Vibration (gravit) (Raw)': 0.5,
            'Discharge Pressure (psi) (Raw)': 1000.0,
            'Virtual Rate (BFPD) (Raw)': 2000.0,
            'Slope': 0.0
        }
        
        # Add any missing columns with default values
        for col, default in default_values.items():
            if col not in df.columns:
                df[col] = default
                logger.warning(f"Added missing column '{col}' with default value {default}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, target_model: str) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # First ensure all required columns exist
        df = self._handle_missing_columns(df, target_model)
        
        # Forward fill for time series data
        df = df.ffill()
        
        # If there are still missing values, fill with column mean or default
        for col in FEATURE_COLUMNS[target_model]:
            if col in df.columns and df[col].isna().any():
                df[col].fillna(df[col].mean(), inplace=True)
        
        return df
    
    def _feature_engineering(self, df: pd.DataFrame, target_model: str) -> pd.DataFrame:
        """Perform feature engineering specific to each model."""
        # Add time-based features
        if 'Reading Time' in df.columns:
            df['hour'] = df['Reading Time'].dt.hour
            df['day_of_week'] = df['Reading Time'].dt.dayofweek
            df['month'] = df['Reading Time'].dt.month
        
        # Add rolling statistics for time series data
        for col in FEATURE_COLUMNS[target_model]:
            if col in df.columns:
                df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24, min_periods=1).mean()
                df[f'{col}_rolling_std_24h'] = df[col].rolling(window=24, min_periods=1).std()
        
        return df
    
    def load_model(self, model_name: str):
        """
        Load a pre-trained model from disk.
        
        Args:
            model_name: Name of the model to load
        """
        model_path = MODEL_PATHS[model_name]
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading {model_name} model from {model_path}")
        self.models[model_name] = load(model_path)
        return self.models[model_name]
    
    def _safe_predict(self, model, X):
        """Make predictions while handling scikit-learn version differences and missing values."""
        from sklearn.impute import SimpleImputer
        
        try:
            # Convert to DataFrame if it's a numpy array and we have feature names
            if isinstance(X, np.ndarray) and hasattr(model, 'feature_names_in_'):
                X = pd.DataFrame(X, columns=model.feature_names_in_)
            
            # Handle missing values safely (avoid shape mismatch when a column is all-NaN)
            if isinstance(X, pd.DataFrame) and X.isna().any().any():
                logger.warning("Input contains NaN values. Imputing with column means.")
                cols_with_nan = X.columns[X.isna().any()].tolist()
                if cols_with_nan:
                    cols_all_nan = [c for c in cols_with_nan if X[c].isna().all()]
                    cols_some_nan = [c for c in cols_with_nan if c not in cols_all_nan]
                    # Fill all-NaN columns with 0.0 (or a safe default)
                    for c in cols_all_nan:
                        X[c] = 0.0
                    if cols_some_nan:
                        imputer = SimpleImputer(strategy='mean')
                        X_imputed = imputer.fit_transform(X[cols_some_nan])
                        X.loc[:, cols_some_nan] = X_imputed
            
            # Handle feature name validation
            try:
                return model.predict(X)
            except (ValueError, AttributeError) as e:
                if "feature names" in str(e).lower():
                    logger.warning("Bypassing feature name validation due to version mismatch")
                    if hasattr(X, 'values'):
                        X = X.values
                    return model.predict(X)
                raise
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Return array of zeros as fallback with the correct length
            if hasattr(X, 'shape') and len(X.shape) > 0:
                return np.zeros(X.shape[0])
            elif hasattr(X, '__len__'):
                return np.zeros(len(X))
            return np.array([0])

    def predict(self, model_name: str, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Make predictions using a loaded model.
        
        Args:
            model_name: Name of the model to use for prediction
            data: Data to make predictions on. If None, uses the loaded data.
            
        Returns:
            np.ndarray: Model predictions
        """
        if model_name not in self.models:
            self.load_model(model_name)
            
        model = self.models[model_name]
        
        # Use provided data or the loaded data
        if data is None:
            if self.data is None:
                raise ValueError("No data provided and no data loaded. Call load_data() first.")
            data = self.data
            
        # Get features for the model
        features = self._get_features(data, model_name)
        
        # Debug: Print model's expected features if available
        if hasattr(model, 'feature_names_in_'):
            logger.debug(f"Model {model_name} expects features: {model.feature_names_in_}")
        elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_names_in_'):
            logger.debug(f"Model {model_name} best estimator expects features: {model.best_estimator_.feature_names_in_}")
        
        logger.info(f"Making predictions with {model_name}")
        logger.debug(f"Provided features: {features.columns.tolist()}")
        
        # Make predictions
        predictions = self._safe_predict(model, features)
        
        # Store predictions
        self.predictions[model_name] = predictions
        
        return predictions
    
    def _get_features(self, data: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Extract features for a specific model, handling missing columns and name mapping.
        
        Args:
            data: Input DataFrame containing the data
            model_name: Name of the model to get features for
            
        Returns:
            DataFrame with the required features in the expected order
        """
        # For failure prediction, we need to map our columns to the model's expected features
        if model_name == 'failure_prediction':
            # Prefer the model's declared feature order if available (short codes from slopes)
            model_feature_names = None
            if model_name in self.models:
                m = self.models[model_name]
                if hasattr(m, 'feature_names_in_'):
                    model_feature_names = list(m.feature_names_in_)
                elif hasattr(m, 'best_estimator_') and hasattr(m.best_estimator_, 'feature_names_in_'):
                    model_feature_names = list(m.best_estimator_.feature_names_in_)

            # If df11 (slopes) is passed in 'data', use those columns directly by model order
            if model_feature_names is not None and all(c in data.columns for c in model_feature_names):
                features = data[model_feature_names].copy()
                logger.debug(f"failure_prediction using model feature order: {model_feature_names}")
            else:
                # Fallback to the canonical slope set
                slope_short = ['A', 'IP', 'DP', 'IT', 'MT', 'V', 'R']
                available = [c for c in slope_short if c in data.columns]
                if not available:
                    # Map from raw to short if slopes not provided
                    feature_mapping = {
                        'A': 'Average Amps (A) (Raw)',
                        'IP': 'Intake Pressure (psi) (Raw)',
                        'DP': 'Discharge Pressure (psi) (Raw)',
                        'IT': 'Intake Temperature (F) (Raw)',
                        'MT': 'Motor Temperature (F) (Raw)',
                        'V': 'Vibration (gravit) (Raw)',
                        'R': 'Virtual Rate (BFPD) (Raw)'
                    }
                    features = pd.DataFrame()
                    for k, src in feature_mapping.items():
                        if src in data.columns:
                            features[k] = data[src]
                        else:
                            features[k] = 0.0
                            logger.warning(f"Filled missing column '{src}' with zeros for failure_prediction")
                    order = model_feature_names if model_feature_names is not None else slope_short
                    missing = [c for c in order if c not in features.columns]
                    for c in missing:
                        features[c] = 0.0
                    features = features[order]
                else:
                    order = model_feature_names if model_feature_names is not None else slope_short
                    # Use only the intersection, preserving model order
                    use_cols = [c for c in order if c in data.columns]
                    features = data[use_cols].copy()
                    # Backfill any missing expected cols with zeros to avoid predict errors
                    for c in order:
                        if c not in features.columns:
                            features[c] = 0.0
                    features = features[order]

        else:
            # If the model carries its own feature names (e.g., short names 'A','IP','IT','MT'), respect that
            model_feature_names = None
            if model_name in self.models:
                m = self.models[model_name]
                if hasattr(m, 'feature_names_in_'):
                    model_feature_names = list(m.feature_names_in_)
                elif hasattr(m, 'best_estimator_') and hasattr(m.best_estimator_, 'feature_names_in_'):
                    model_feature_names = list(m.best_estimator_.feature_names_in_)

            short_set = {'A', 'IP', 'DP', 'IT', 'MT', 'V', 'R'}
            if model_feature_names is not None and set(model_feature_names).issubset(short_set):
                # Build features by mapping from raw columns to short names
                mapping = {
                    'A': 'Average Amps (A) (Raw)',
                    'IP': 'Intake Pressure (psi) (Raw)',
                    'DP': 'Discharge Pressure (psi) (Raw)',
                    'IT': 'Intake Temperature (F) (Raw)',
                    'MT': 'Motor Temperature (F) (Raw)',
                    'V': 'Vibration (gravit) (Raw)',
                    'R': 'Virtual Rate (BFPD) (Raw)'
                }
                cols = []
                for k in model_feature_names:
                    src = mapping.get(k)
                    if src == 'Discharge Pressure (psi) (Raw)' and src not in data.columns and 'predicted_discharge_pressure' in data.columns:
                        series = data['predicted_discharge_pressure']
                        logger.info("Using predicted_discharge_pressure for DP feature")
                    else:
                        if src in data.columns:
                            series = data[src]
                        else:
                            logger.warning(f"Filled missing column '{src}' with zeros for model {model_name}")
                            series = pd.Series(np.zeros(len(data)), index=data.index)
                    cols.append(series)
                features = pd.concat(cols, axis=1)
                features.columns = model_feature_names
            else:
                # Fallback to FEATURE_COLUMNS configuration
                required_cols = FEATURE_COLUMNS[model_name]

                # Log available and required columns for debugging
                logger.debug(f"Available columns in data: {data.columns.tolist()}")
                logger.debug(f"Required columns for {model_name}: {required_cols}")

                missing_cols = [col for col in required_cols if col not in data.columns]

                if missing_cols:
                    logger.warning(f"Missing columns for {model_name}: {missing_cols}. Using available columns.")
                    available_cols = [col for col in required_cols if col in data.columns]
                    if not available_cols:
                        raise ValueError(f"No required columns found in data for {model_name}")
                    logger.info(f"Using available columns for {model_name}: {available_cols}")
                    features = data[available_cols].copy()

                    # Fill missing columns; for virtual_rate, inject predicted DP if needed
                    for col in missing_cols:
                        if (
                            model_name == 'virtual_rate'
                            and col == 'Discharge Pressure (psi) (Raw)'
                            and 'predicted_discharge_pressure' in data.columns
                        ):
                            features[col] = data['predicted_discharge_pressure']
                            logger.info("Filled missing 'Discharge Pressure (psi) (Raw)' with predicted_discharge_pressure for virtual_rate")
                        else:
                            features[col] = 0.0
                            logger.warning(f"Filled missing column '{col}' with zeros")
                else:
                    # All required columns are present
                    features = data[required_cols].copy()

                # Ensure the column order matches the expected order
                features = features[required_cols]
        
        # Log feature information
        logger.debug(f"Final feature columns for {model_name}: {features.columns.tolist()}")
        logger.debug(f"Feature data types: {features.dtypes}")
        if len(features) > 0:
            logger.debug(f"First row of features: {features.iloc[0].to_dict()}")
        
        return features
    def run_full_analysis(self, input_file: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """
        Run the complete analysis pipeline for the well in the following order:
        1. Discharge Pressure Prediction
        2. Virtual Rate Prediction
        3. Slope Calculation
        4. Failure Prediction

        Args:
            input_file: Path to the input CSV file. If None, looks in the default input directory.

        Returns:
            Dict containing all predictions and calculations
        """
        try:
            # Load the data if not already loaded
            if self.data is None or self.data.empty:
                self.load_data(input_file)

            # Ensure timestamp is datetime and sorted
            if 'Reading Time' in self.data.columns:
                self.data['Reading Time'] = pd.to_datetime(self.data['Reading Time'], format='mixed', dayfirst=False, errors='coerce')
                self.data = self.data.sort_values('Reading Time')

            results = {}

            # Try to load daily Watercut data from prod_data.csv if available
            try:
                self._load_wc_data()
            except Exception as e:
                logger.warning(f"Could not load Watercut data: {e}")

            # 1. Discharge Pressure Prediction (predicted_discharge_pressure)
            logger.info("1/4 - Running discharge pressure prediction...")
            dp_col = 'Discharge Pressure (psi) (Raw)'
            has_dp_col = dp_col in self.data.columns
            dp_series = self.data[dp_col] if has_dp_col else pd.Series(dtype=float)
            dp_all_nan = (not has_dp_col) or dp_series.isna().all()

            discharge_pressure = self.predict('discharge_pressure')
            results['discharge_pressure'] = discharge_pressure

            # Always retain the raw DP column if it already exists
            if not has_dp_col:
                self.data[dp_col] = np.nan

            # Store predicted values for auditing
            self.data['predicted_discharge_pressure'] = discharge_pressure

            # Only overwrite the raw column when it matches the notebook condition (all NaN)
            if dp_all_nan:
                logger.info("DP column entirely NaN → overwriting with model predictions to mirror notebook behavior")
                self.data[dp_col] = discharge_pressure

            # Export intermediate file just like the notebook
            self._export_notebook_style_csv('SKW Final.csv', self.data)
            # Also save to well-specific output folder
            try:
                well_skw_final = self.output_dir / 'SKW Final.csv'
                self.data.drop(columns=[c for c in self.data.columns if c.startswith('predicted_')], errors='ignore').to_csv(well_skw_final, index=False)
            except Exception as e:
                logger.warning(f'Could not save well-specific SKW Final.csv: {e}')

            # 2. Virtual Rate: replicate notebook exactly (dropna before predict, zero rule)
            logger.info("2/4 - Virtual rate handling...")
            vr_col = 'Virtual Rate (BFPD) (Raw)'

            # Notebook line 179: df.dropna(inplace=True)
            data_before_dropna = self.data.copy()
            self.data = self.data.dropna().reset_index(drop=True)
            logger.info(f"Dropped NaN rows before VR prediction: {len(data_before_dropna)} -> {len(self.data)} rows")

            # Predict VR exactly as notebook
            logger.info("Running Virtual Rate prediction (notebook behavior)")
            virtual_rate = self.predict('virtual_rate')
            results['virtual_rate'] = virtual_rate
            self.data[vr_col] = virtual_rate

            if {'Average Amps (A) (Raw)', 'Drive Frequency (Hz) (Raw)'}.issubset(self.data.columns):
                logger.info("Applying zero rule: if Amps==0 and Freq==0 then VR=0")
                self.data[vr_col] = self.data.apply(
                    lambda row: 0 if (row['Average Amps (A) (Raw)'] == 0 and row['Drive Frequency (Hz) (Raw)'] == 0)
                    else row[vr_col],
                    axis=1
                )

            # Export identical to notebook
            self._export_notebook_style_csv('SKW_final_w_Pd.csv', self.data)
            # Also save to well-specific output folder (without predicted_ columns)
            try:
                well_skw_vr = self.output_dir / 'SKW_final_w_Pd.csv'
                export_df = self.data.drop(columns=[c for c in self.data.columns if c.startswith('predicted_')], errors='ignore')
                export_df.to_csv(well_skw_vr, index=False)
            except Exception as e:
                logger.warning(f'Could not save well-specific SKW_final_w_Pd.csv: {e}')

            # Prepare dataframe exactly like notebook (df1)
            df1 = self.data.copy().reset_index(drop=True)
            if 'predicted_discharge_pressure' in df1.columns:
                df1 = df1.drop(columns=['predicted_discharge_pressure'])

            # Notebook drops column index 4 before slope calc
            df_for_slopes = df1.drop(df1.columns[4], axis=1)

            # Compute df_all (30-minute resample) identical to notebook code
            logger.info("3/4 - Building 30-minute resampled dataset (df_all)...")
            df_all = df1.copy()
            df_all['Reading Time'] = pd.to_datetime(df_all['Reading Time'], format='mixed', dayfirst=False, errors='coerce')
            df_all = (
                df_all.set_index('Reading Time')
                      .resample('30min', origin='epoch')
                      .mean(numeric_only=True)
                      .reset_index()
            )

            # Save df_all exactly as notebook
            try:
                project_root = Path(__file__).resolve().parents[1]
                df_all_path = project_root / 'df_all.csv'
                df_all.to_csv(df_all_path, index=False)
                logger.info(f"Saved df_all to: {df_all_path}")
                # Also save to well-specific folder
                well_df_all = self.output_dir / 'df_all.csv'
                df_all.to_csv(well_df_all, index=False)
            except Exception as e:
                logger.warning(f"Could not save df_all.csv: {e}")

            # 4. Compute slopes exactly as notebook
            logger.info("4/4 - Computing 30-minute window slopes...")
            df_for_slopes = df_for_slopes.copy()
            df_for_slopes['Reading Time'] = pd.to_datetime(df_for_slopes['Reading Time'], format='mixed', dayfirst=False, errors='coerce')
            df_for_slopes = df_for_slopes.dropna(subset=['Reading Time']).sort_values('Reading Time').reset_index(drop=True)
            dup_groups = df_for_slopes.groupby('Reading Time').cumcount()
            df_for_slopes['Reading Time'] = df_for_slopes['Reading Time'] + pd.to_timedelta(dup_groups * 30, unit='s')
            df_for_slopes = df_for_slopes.sort_values('Reading Time').reset_index(drop=True)

            if df_for_slopes.empty:
                slopes_df = pd.DataFrame(columns=['Window_Start_Time','A','IP','DP','IT','MT','V','R'])
            else:
                # Vectorized slope computation per 30-min window using OLS formula
                # Define required raw columns for slopes (match notebook mapping)
                raw_cols_map = [
                    ('Average Amps (A) (Raw)', 'A'),
                    ('Intake Pressure (psi) (Raw)', 'IP'),
                    ('Discharge Pressure (psi) (Raw)', 'DP'),
                    ('Intake Temperature (F) (Raw)', 'IT'),
                    ('Motor Temperature (F) (Raw)', 'MT'),
                    ('Vibration (gravit) (Raw)', 'V'),
                    ('Virtual Rate (BFPD) (Raw)', 'R'),
                ]
                present_raw = [c for c, _ in raw_cols_map if c in df_for_slopes.columns]

                df_s = df_for_slopes[['Reading Time'] + present_raw].copy()
                df_s['Window_Start_Time'] = df_s['Reading Time'].dt.floor('30min')

                # Only keep windows with at least 2 rows (match loop behavior that skips <2)
                win_counts = df_s.groupby('Window_Start_Time')['Reading Time'].size()
                valid_windows = set(win_counts[win_counts >= 2].index)
                if len(valid_windows) == 0:
                    slopes_df = pd.DataFrame(columns=['Window_Start_Time','A','IP','DP','IT','MT','V','R'])
                else:
                    df_s = df_s[df_s['Window_Start_Time'].isin(valid_windows)]
                    df_s['tsec'] = (df_s['Reading Time'] - df_s['Window_Start_Time']).dt.total_seconds()

                    # Long form for vectorized aggregation
                    long = df_s.melt(
                        id_vars=['Window_Start_Time', 'tsec'],
                        value_vars=present_raw,
                        var_name='metric',
                        value_name='value'
                    ).dropna(subset=['value'])

                    if long.empty:
                        slopes_df = pd.DataFrame(columns=['Window_Start_Time','A','IP','DP','IT','MT','V','R'])
                    else:
                        long['ty'] = long['tsec'] * long['value']
                        long['t2'] = long['tsec'] * long['tsec']
                        agg = long.groupby(['Window_Start_Time', 'metric']).agg(
                            n=('value', 'count'),
                            sum_t=('tsec', 'sum'),
                            sum_y=('value', 'sum'),
                            sum_ty=('ty', 'sum'),
                            sum_t2=('t2', 'sum')
                        ).reset_index()

                        n = agg['n'].to_numpy(dtype=float)
                        sum_t = agg['sum_t'].to_numpy(dtype=float)
                        sum_y = agg['sum_y'].to_numpy(dtype=float)
                        sum_ty = agg['sum_ty'].to_numpy(dtype=float)
                        sum_t2 = agg['sum_t2'].to_numpy(dtype=float)

                        den = sum_t2 - (sum_t * sum_t) / n
                        with np.errstate(divide='ignore', invalid='ignore'):
                            slope_vals = (sum_ty - (sum_t * sum_y) / n) / den
                        slope_vals[(n < 2) | ~np.isfinite(slope_vals)] = np.nan
                        agg['slope'] = slope_vals

                        pivot = agg.pivot(index='Window_Start_Time', columns='metric', values='slope').reset_index()
                        # Build final slopes_df with expected columns
                        slopes_df = pivot[['Window_Start_Time']].copy()
                        for raw_name, short in raw_cols_map:
                            if raw_name in pivot.columns:
                                slopes_df[short] = pivot[raw_name]
                            else:
                                slopes_df[short] = np.nan

            # Prepare df11 features
            expected_cols = ['A', 'IP', 'DP', 'IT', 'MT', 'V', 'R']
            df11 = slopes_df[expected_cols].copy() if not slopes_df.empty else pd.DataFrame(columns=expected_cols)

            # Save slopes/X_predict exactly like notebook
            try:
                project_root = Path(__file__).resolve().parents[1]
                slopes_path = project_root / 'slopes_df_30menit.csv'
                slopes_df.to_csv(slopes_path, index=False)
                logger.info(f"Saved slopes_df_30menit to: {slopes_path}")

                xpred_path = project_root / 'X_predict_30menit.csv'
                df11.to_csv(xpred_path, index=False)
                logger.info(f"Saved X_predict_30menit to: {xpred_path}")
                
                # Also save to well-specific folder
                well_slopes = self.output_dir / 'slopes_df_30menit.csv'
                slopes_df.to_csv(well_slopes, index=False)
                well_xpred = self.output_dir / 'X_predict_30menit.csv'
                df11.to_csv(well_xpred, index=False)
            except Exception as e:
                logger.warning(f"Could not save slope feature CSVs: {e}")

            # 5. Failure Prediction using df11 (exact notebook behavior)
            if df11 is None or len(df11) == 0:
                logger.warning("No slope feature rows available; using zeros for failure prediction output")
                failure_pred = np.zeros(len(slopes_df), dtype=int)
            else:
                failure_pred = self.predict('failure_prediction', data=df11)
            results['failure_prediction'] = failure_pred

            # Assemble final result DataFrame to match template
            # Ensure predictions match number of windows
            if len(failure_pred) != len(slopes_df):
                logger.warning(
                    f"Prediction length {len(failure_pred)} does not match windows {len(slopes_df)}; adjusting"
                )
                if len(slopes_df) == 0:
                    failure_pred = np.array([], dtype=int)
                elif len(failure_pred) == 0:
                    failure_pred = np.zeros(len(slopes_df), dtype=int)
                elif len(failure_pred) > len(slopes_df):
                    failure_pred = np.asarray(failure_pred)[:len(slopes_df)]
                else:
                    # pad with zeros
                    pad = np.zeros(len(slopes_df) - len(failure_pred), dtype=int)
                    failure_pred = np.concatenate([np.asarray(failure_pred), pad])

            final_df = self._assemble_failure_results(slopes_df, df_all, failure_pred)

            # Save joined debug CSV for auditing: features + predictions/status
            try:
                project_root = Path(__file__).resolve().parents[1]
                features_used_path = project_root / 'failure_features_used_30menit.csv'
                if features_used_path.exists():
                    fu = pd.read_csv(features_used_path)
                    joined = fu.merge(final_df[['Window_Start_Time','Prediction','Status']], on='Window_Start_Time', how='left')
                    audit_path = project_root / 'prediction_with_features_30menit.csv'
                    joined.to_csv(audit_path, index=False)
                    logger.info(f"Saved prediction_with_features_30menit to: {audit_path}")
            except Exception as e:
                logger.warning(f"Could not save prediction_with_features_30menit.csv: {e}")

            # Save outputs
            self._save_results(results)  # original simple CSVs
            self._save_failure_results(final_df)  # template-like final output
            
            # ============================================================
            # 6. Tambahkan kolom Indicator dan generate latest report
            # ============================================================
            try:
                logger.info("5/4 - Adding Indicator columns and generating latest report...")
                
                # Merge slopes ke final_df untuk keperluan indicator
                final_df_with_slopes = final_df.copy()
                if not slopes_df.empty and 'Window_Start_Time' in slopes_df.columns:
                    # Merge slopes (A, IP, DP, IT, MT, V, R)
                    slope_cols = ['A', 'IP', 'DP', 'IT', 'MT', 'V', 'R']
                    merge_cols = ['Window_Start_Time'] + [c for c in slope_cols if c in slopes_df.columns]
                    final_df_with_slopes = final_df_with_slopes.merge(
                        slopes_df[merge_cols], 
                        on='Window_Start_Time', 
                        how='left'
                    )
                
                # Load 3-hour results yang sudah disimpan
                output_dir = str(self.output_dir)
                result_3h_file = os.path.join(output_dir, "result_df_3 jam.csv")
                if os.path.exists(result_3h_file):
                    result_3h = pd.read_csv(result_3h_file)
                    result_3h['Window_Start_Time'] = pd.to_datetime(result_3h['Window_Start_Time'])
                    
                    # Tambahkan indicators ke kedua dataset
                    indicator_30min, result_3h_with_ind = self._add_indicators_to_results(
                        final_df_with_slopes, 
                        result_3h
                    )
                    
                    # Save hasil dengan indicator
                    # 30 menit dengan indicator
                    indicator_30min_file = os.path.join(output_dir, f"{self.well_name}_indicator_30min.csv")
                    indicator_30min.to_csv(indicator_30min_file, index=False)
                    logger.info(f"Saved 30-minute results with indicators to: {indicator_30min_file}")
                    
                    # 3 jam dengan indicator
                    result_3h_ind_file = os.path.join(output_dir, "result_df_3jam_with_indicator.csv")
                    result_3h_with_ind.to_csv(result_3h_ind_file, index=False)
                    logger.info(f"Saved 3-hour results with indicators to: {result_3h_ind_file}")

                    # Overwrite result_df_3 jam.csv (well folder + root) dengan versi ber-Indicator,
                    # agar persis seperti output notebook new.py
                    try:
                        result_3h_main_file = os.path.join(output_dir, "result_df_3 jam.csv")
                        result_3h_with_ind.to_csv(result_3h_main_file, index=False)
                        logger.info(f"Overwrote 3-hour aggregated results with indicators at: {result_3h_main_file}")

                        project_root = Path(__file__).resolve().parents[1]
                        root_file = project_root / 'result_df_3 jam.csv'
                        result_3h_with_ind.to_csv(root_file, index=False)
                        logger.info(f"Overwrote 3-hour aggregated results at project root: {root_file}")
                    except Exception as e:
                        logger.warning(f"Could not overwrite 3-hour results with indicators: {e}")
                    
                    # Generate latest report data untuk Flask UI
                    logger.info("Generating Latest Status and Latest Failure data for UI...")
                    latest_report_data = self._generate_latest_report(indicator_30min, result_3h_with_ind)
                    
                    # Save report data as JSON untuk Flask
                    import json
                    report_json_file = os.path.join(output_dir, f"{self.well_name}_latest_report.json")
                    with open(report_json_file, 'w', encoding='utf-8') as f:
                        json.dump(latest_report_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved latest report data to: {report_json_file}")
                    
                    # Store in results for return to Flask
                    results['latest_report'] = latest_report_data
                    
                else:
                    logger.warning("3-hour results file not found, skipping indicator generation")
                    
            except Exception as e:
                logger.warning(f"Could not add indicators or generate report: {e}")

            logger.info("Pipeline finished with template-aligned outputs.")
            return results

        except Exception as e:
            logger.error(f"Error during pipeline execution: {str(e)}", exc_info=True)
            raise
    
    def calculate_slopes(self, window_minutes: int = 30) -> np.ndarray:
        """
        Calculate slopes for the virtual rate time series data.
        
        Args:
            window_minutes: Size of the rolling window in minutes
            
        Returns:
            Array of slope values
        """
        try:
            # Make sure we have the required columns
            required_cols = ['predicted_virtual_rate', 'Reading Time']
            if not all(col in self.data.columns for col in required_cols):
                logger.warning("Missing required columns for slope calculation. Returning zeros.")
                return np.zeros(len(self.data))
            
            # Create a working copy and ensure proper data types
            df = self.data[required_cols].copy()
            
            # Ensure timestamp is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(df['Reading Time']):
                df['Reading Time'] = pd.to_datetime(df['Reading Time'], format='mixed', dayfirst=False, errors='coerce')
            
            # Sort by time
            df = df.sort_values('Reading Time')
            
            # Calculate time differences in minutes
            time_diff = df['Reading Time'].diff().dt.total_seconds().fillna(0) / 60.0
            
            # Calculate rate differences
            rate_diff = df['predicted_virtual_rate'].diff().fillna(0)
            
            # Calculate slopes (rate change per minute)
            with np.errstate(divide='ignore', invalid='ignore'):
                # Only calculate slope if time difference is positive
                valid_mask = (time_diff > 0) & (time_diff <= 1440)  # Max 1 day difference
                slopes = np.zeros(len(df))
                slopes[valid_mask] = rate_diff[valid_mask] / time_diff[valid_mask]
            
            # Handle any remaining invalid values
            slopes = np.nan_to_num(slopes, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply rolling window average if specified
            if window_minutes > 0:
                window_size = max(1, int(window_minutes / (time_diff[1:].median() or 1)))
                if window_size > 1:
                    slopes = np.convolve(slopes, np.ones(window_size)/window_size, mode='same')
            
            return slopes
            
        except Exception as e:
            logger.error(f"Error calculating slopes: {str(e)}", exc_info=True)
            return np.zeros(len(self.data))

    def _build_df_all_30min(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample numeric columns to 30-minute intervals aligned to :00 and :30.
        Returns a DataFrame with 'Reading Time' as the resampled timestamp.
        """
        if 'Reading Time' not in df.columns:
            raise ValueError("'Reading Time' column is required for resampling")

        df_idx = df.copy()
        df_idx['Reading Time'] = pd.to_datetime(df_idx['Reading Time'], format='mixed', dayfirst=False, errors='coerce')
        df_idx = df_idx.dropna(subset=['Reading Time']).set_index('Reading Time')
        # Align to :00 and :30 using origin='epoch' to snap to half-hour grid
        df_resampled = (
            df_idx.resample('30min', label='left', closed='left', origin='epoch')
                 .mean(numeric_only=True)
                 .reset_index()
        )
        # Ensure column name matches template
        if 'Reading Time' not in df_resampled.columns:
            df_resampled = df_resampled.rename(columns={'index': 'Reading Time'})
        return df_resampled

    def _compute_window_slopes_30min(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        - Build explicit windows with pd.date_range aligned to :00/:30
        - Use seconds since window start and scipy linregress
        - Skip windows with <2 points
        Returns: DataFrame with ['Window_Start_Time','A','IP','DP','IT','MT','V','R']
        """
        if 'Reading Time' not in df.columns:
            raise ValueError("'Reading Time' column is required for slope computation")

        from scipy.stats import linregress

        use_cols_map = {
            'A': 'Average Amps (A) (Raw)',
            'IP': 'Intake Pressure (psi) (Raw)',
            'DP': 'Discharge Pressure (psi) (Raw)',
            'IT': 'Intake Temperature (F) (Raw)',
            'MT': 'Motor Temperature (F) (Raw)',
            'V': 'Vibration (gravit) (Raw)',
            'R': 'Virtual Rate (BFPD) (Raw)',
        }

        d = df.copy()
        d['Reading Time'] = pd.to_datetime(d['Reading Time'], format='mixed', dayfirst=False, errors='coerce')
        d = d.dropna(subset=['Reading Time']).sort_values('Reading Time').reset_index(drop=True)
        if d.empty:
            return pd.DataFrame(columns=['Window_Start_Time'] + list(use_cols_map.keys()))

        start_time = d['Reading Time'].iloc[0].floor('30min')
        end_time = d['Reading Time'].iloc[-1].ceil('30min')
        windows = pd.date_range(start=start_time, end=end_time, freq='30min')

        rows = []
        for w_start in windows:
            w_end = w_start + pd.Timedelta(minutes=30)
            w_df = d[(d['Reading Time'] >= w_start) & (d['Reading Time'] < w_end)]
            if len(w_df) < 2:
                # Still record the window with NaNs (consistent with skipping in notebook exports)
                row = {k: np.nan for k in use_cols_map.keys()}
                row['Window_Start_Time'] = w_start
                rows.append(row)
                continue

            # seconds since window start
            tsec = (w_df['Reading Time'] - w_start).dt.total_seconds().to_numpy()
            row = {'Window_Start_Time': w_start}
            for short, col in use_cols_map.items():
                if col not in w_df.columns:
                    row[short] = np.nan
                    continue
                y = w_df[col].to_numpy(dtype=float)
                try:
                    slope, _, _, _, _ = linregress(tsec, y)
                except Exception:
                    slope = np.nan
                row[short] = slope
            rows.append(row)

        out = pd.DataFrame(rows)
        return out

    def _assemble_failure_results(
        self,
        slopes_df: pd.DataFrame,
        df_all: pd.DataFrame,
        predictions: np.ndarray,
    ) -> pd.DataFrame:
        """Create final failure prediction DataFrame with status and recommendation,
        applying additional rules for Shut-in and EDP similar to the template logic.
        """
        # Base frame
        out = pd.DataFrame({
            'Window_Start_Time': slopes_df['Window_Start_Time'],
            'Prediction': predictions.astype(int) if predictions is not None else 0,
        })

        # Mapping functions
        def status_map(x: int) -> str:
            x = int(x)
            return {
                0: 'Running',
                1: 'Low PI',
                2: 'Pump Wear',
                3: 'Tubing Leak',
                4: 'Higher PI',
                5: 'Increase in Frequency',
                6: 'Open Choke',
                7: 'Increase in Watercut',
                8: 'Sand Ingestion',
                9: 'Closed Valve',
                10: 'Electrical Downhole Problem',
                11: 'Shut-in',
                12: '100% Watercut',
                13: 'Start-up Phase',
            }.get(x, 'Unidentified')

        def recommendation_map(x: int) -> str:
            """Match notebook's recommendation() function exactly.
            Notebook only returns values for x==0 (' ') and else ('Unidentified').
            For x in 1-13, it prints to console but returns None (saved as empty/NaN in CSV).
            """
            x = int(x)
            if x == 0:
                return " "
            recs = {
                1: (
                    "The Possibility Causes: Well productivity less than pump design range\n"
                    "1. Analyze fluid level and Bottom Hole Pressure (BHP)\n"
                    "2. Adjust tubing wellhead pressure to bring pump rate within design rate\n"
                    "The Possibility Causes: Restricted pump\n"
                    "1. Inject solvent or diluent through annulus if fluid is highly viscous\n"
                    "2. Use VSD “rocking mode” to remove debris"
                ),
                2: (
                    "1. Check performance drop (>15–20% from initial installation)\n"
                    "2. Verify vibration increase (>20%)\n"
                    "3. Perform shut-in test with surface check valve closed while pump is running"
                ),
                3: (
                    "1. Confirm by a pressure test at the tubing wellhead\n"
                    "2. Meanwhile, fill up the tubing and pressure up against RCV"
                ),
                4: (
                    "The Possibility Causes: Well productivity above pump design range\n"
                    "1. Analyze fluid level and BHP\n"
                    "2. Adjust wellhead pressure to maintain design rate\n"
                    "The Possibility Causes: Change in fluid characteristics\n"
                    "1. Analyze fluid level and BHP\n"
                    "2. Conduct fluid analysis for pump re-design reference"
                ),
                5: (
                    "1. Compare discharge pressure with historical data\n"
                    "2. Reduce frequency via VSD"
                ),
                6: (
                    "Check pump discharge pressure and production rate, compare with historical well data"
                ),
                7: (
                    "Adjust tubing wellhead pressure to bring production rate within design limits\n"
                ),
                8: (
                    "1. Check flow line and separator for evidence of sand, mud, or debris\n"
                    "2. Design solid control system for next installation"
                ),
                9: (
                    "1. Verify if the valve was deliberately partially closed by Field Service Tech\n"
                    "2. Contact the Field Technician for on-site inspection"
                ),
                10: (
                    "1. Verify surface equipment (VSD, transformer, junction box) to isolate downhole issue\n"
                    "2. Perform VSD soft shutdown to prevent reverse current damage\n"
                    "3. Conduct a DIFA (Dismantle Inspection and Failure Analysis"
                ),
                11: (
                    "Shut-in detected. Verify operating schedule and surface conditions. Ensure Amps/Frequency are expected to be zero."
                ),
                12: (
                    "The Possibility Causes: Well producing 100% water—possible water breakthrough or reservoir depletion\n"
                    "1. Check production test and GOR trend to confirm water source\n"
                    "2. Inspect well completion for leaks; consider shut-in, isolation, or water-shutoff treatment"
                ),
                13: (
                    "Start-up Phase after extended data gap:\n"
                    "1. Conduct no-load test before commissioning to ensure proper drive operation\n"
                    "2. Verify drive parameters, transformer taps, and gauge units are correctly set\n"
                    "3. Test both rotations; select the one with lower amperage (correct pump rotation)\n"
                    "4. Monitor WHP—if no buildup, briefly choke the well to initiate flow and confirm ΔP"
                ),
            }

            return recs.get(x, " ")
        
        out['Status'] = out['Prediction'].apply(status_map)
        out['Recommendation'] = out['Prediction'].apply(recommendation_map)
        # Ensure Recommendation is always a string (notebook uses ' ' not NaN)
        out['Recommendation'] = out['Recommendation'].fillna(' ').astype(str)
        out['Recommendation'] = (
            out['Recommendation']
            .str.replace('\n', ' ', regex=False)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )
        # Replace empty strings with single space to match notebook
        out.loc[out['Recommendation'] == '', 'Recommendation'] = ' '

        # Apply additional rules using vectorized operations
        try:
            TOL = 1e-8
            df_all2 = df_all.copy()
            df_all2['Reading Time'] = pd.to_datetime(df_all2['Reading Time'], format='mixed', dayfirst=False, errors='coerce')
            slopes_df2 = slopes_df.copy()
            slopes_df2['Window_Start_Time'] = pd.to_datetime(slopes_df2['Window_Start_Time'], errors='coerce')

            # Merge out with df_all (resampled point at window start)
            merged = (out
                      .merge(df_all2, left_on='Window_Start_Time', right_on='Reading Time', how='left', suffixes=('', '_res'))
                      .merge(slopes_df2, on='Window_Start_Time', how='left', suffixes=('', '_slope')))

            # Watercut override: vectorized per notebook (daily WC table)
            mask_wc = pd.Series(False, index=merged.index)
            if hasattr(self, 'df_wc') and self.df_wc is not None and not self.df_wc.empty:
                wc_source = self.df_wc.copy()
                if 'Date' not in wc_source.columns:
                    logger.warning("Watercut DataFrame missing 'Date' column; skipping 100% WC override")
                else:
                    wc_source['Date'] = pd.to_datetime(wc_source['Date'], errors='coerce')
                    wc_source['WC'] = pd.to_numeric(wc_source['WC'], errors='coerce')
                    wc_source = wc_source.dropna(subset=['Date'])
                    merged['Date'] = merged['Window_Start_Time'].dt.normalize()
                    wc_map = wc_source.set_index('Date')['WC']
                    merged['WC_val'] = merged['Date'].map(wc_map)
                    mask_wc = np.isclose(merged['WC_val'], 100.0, atol=1e-6)
                    wc_missing = merged['WC_val'].isna().sum()
                    if wc_missing:
                        logger.debug("WC mapping missing for %s windows (no matching production date)", wc_missing)
            else:
                logger.warning("Watercut data unavailable; 100% Watercut override skipped")

            # Shortcuts for columns (fillna with 0 for comparisons)
            amps = merged.get('Average Amps (A) (Raw)', pd.Series(np.nan, index=merged.index)).fillna(0.0)
            freq = merged.get('Drive Frequency (Hz) (Raw)', pd.Series(np.nan, index=merged.index)).fillna(0.0)
            rate = merged.get('Virtual Rate (BFPD) (Raw)', pd.Series(np.nan, index=merged.index)).fillna(0.0)

            dp_s = merged.get('DP', pd.Series(np.nan, index=merged.index)).fillna(0.0)
            it_s = merged.get('IT', pd.Series(np.nan, index=merged.index)).fillna(0.0)
            mt_s = merged.get('MT', pd.Series(np.nan, index=merged.index)).fillna(0.0)
            v_s = merged.get('V', pd.Series(np.nan, index=merged.index)).fillna(0.0)
            r_s = merged.get('R', pd.Series(np.nan, index=merged.index)).fillna(0.0)

            # EXACT notebook lines 868-920: Cek variasi dari RAW data dalam window, bukan dari slopes
            # Kolom yang dicek untuk variasi (notebook line 869-876)
            cols_check = [
                "Intake Pressure (psi) (Raw)",
                "Discharge Pressure (psi) (Raw)",
                "Intake Temperature (F) (Raw)",
                "Motor Temperature (F) (Raw)",
                "Vibration (gravit) (Raw)",
                "Virtual Rate (BFPD) (Raw)"
            ]
            
            # Vectorized: compute variation per window using raw df mapped to 30-min buckets
            try:
                df_raw = self.data.copy()
                df_raw['Reading Time'] = pd.to_datetime(df_raw['Reading Time'], format='mixed', dayfirst=False, errors='coerce')
                df_raw = df_raw.dropna(subset=['Reading Time'])
                df_raw['Window_Start_Time'] = df_raw['Reading Time'].dt.floor('30min')
                present_cols = [c for c in cols_check if c in df_raw.columns]
                if present_cols:
                    nu = df_raw.groupby('Window_Start_Time')[present_cols].nunique(dropna=True)
                    has_var_by_win = (nu > 1).any(axis=1)
                else:
                    has_var_by_win = pd.Series(dtype=bool)
                any_variation = merged['Window_Start_Time'].map(has_var_by_win).fillna(False).astype(bool)
            except Exception:
                # Fallback to no-variation if any error occurs (conservative)
                any_variation = pd.Series(False, index=merged.index)

            # Shut-in mask - sesuai notebook: hanya cek VR, DP, dan Vibration (bukan semua kolom)
            amps_zero = np.isclose(amps, 0.0, atol=TOL)
            freq_zero = np.isclose(freq, 0.0, atol=TOL)
            other_zero = (
                np.isclose(rate, 0.0, atol=TOL)
                & np.isclose(merged.get('Discharge Pressure (psi) (Raw)', pd.Series(0, index=merged.index)).fillna(0.0), 0.0, atol=TOL)
                & np.isclose(merged.get('Vibration (gravit) (Raw)', pd.Series(0, index=merged.index)).fillna(0.0), 0.0, atol=TOL)
            )
            mask_shutin = amps_zero & freq_zero & (other_zero | any_variation) & (~mask_wc)

            # EDP override: if Amps and Freq are zero AND DP/IP/IT/MT/V and Rate are also zero (no variation)
            mask_edp = (
                amps_zero & freq_zero &
                np.isclose(rate, 0.0, atol=TOL) &
                np.isclose(dp_s, 0.0, atol=TOL) &
                np.isclose(it_s, 0.0, atol=TOL) &
                np.isclose(mt_s, 0.0, atol=TOL) &
                np.isclose(v_s, 0.0, atol=TOL) &
                np.isclose(r_s, 0.0, atol=TOL) &
                (~any_variation) &
                (~mask_shutin) &
                (~mask_wc)
            )

            # Start from current predictions
            pred_vec = merged['Prediction'].astype(int).to_numpy(copy=True)

            # Apply overrides:
            # Untuk menyamai notebook saat ini, hanya Shut-in dan EDP yang dioverride.
            # Watercut=100% (class 12) belum digunakan di notebook SKW-02, jadi
            # kita tidak mengubah Prediction berdasarkan mask_wc.
            pred_vec = np.where(mask_wc, 12, pred_vec)
            pred_vec = np.where(mask_shutin, 11, pred_vec)
            pred_vec = np.where(mask_edp, 10, pred_vec)

            # Log counts (raw mask sums)
            wc_count = int(np.sum(mask_wc))
            shutin_count = int(np.sum(mask_shutin))
            edp_count = int(np.sum(mask_edp))
            logger.info(
                "Override counts (masks) - Watercut=100%: %s, Shut-in: %s, EDP: %s",
                wc_count,
                shutin_count,
                edp_count,
            )

            # Write back into out by aligning indices
            out = out.merge(merged[['Window_Start_Time']], on='Window_Start_Time', how='left')
            out['Prediction'] = pred_vec

            # Date column (extracted from Window_Start_Time, matching notebook)
            # Will be added after status mapping

            # Remap Status/Recommendation after overrides
            out['Status'] = out['Prediction'].apply(status_map)
            out['Recommendation'] = out['Prediction'].apply(recommendation_map)
            out['Recommendation'] = (
                out['Recommendation']
                .astype(str)
                .str.replace('\n', ' ', regex=False)
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
            )

            # Keep all statuses from template: Low PI, Shut-in, 100% Watercut, EDP, Running, etc.
            # No restriction - allow all classes as per template

            # Add Date column (matching notebook: date portion of Window_Start_Time)
            out['Date'] = pd.to_datetime(out['Window_Start_Time']).dt.date

            # --- Start-up Phase Detection - EXACT sesuai notebook ---
            # Notebook deteksi gap dari prediction_results_df (windowed data), bukan raw data
            # Lines 954-993 di notebook
            out = out.sort_values('Window_Start_Time').reset_index(drop=True)
            out['Status'] = out['Status'].str.strip()

            # EXACT notebook lines 957-993: Loop through windowed predictions untuk cari gap
            for i in range(1, len(out)):
                prev_time = out.loc[i-1, 'Window_Start_Time']
                curr_time = out.loc[i, 'Window_Start_Time']
                gap_hours = (curr_time - prev_time).total_seconds() / 3600.0
                
                if gap_hours > 3:  # gap terdeteksi (notebook line 962)
                    # bacaan pertama setelah gap
                    first_after_gap_idx = i
                    first_after_gap_time = curr_time
                    
                    logger.info(f"Gap detected: {gap_hours:.1f}h between {prev_time} and {curr_time}")
                    
                    # cari Shut-in dalam 3 hari setelah first_after_gap_time (notebook lines 967-973)
                    three_days_later = first_after_gap_time + pd.Timedelta(days=3)
                    shutin_indices = out[
                        (out['Window_Start_Time'] >= first_after_gap_time) &
                        (out['Window_Start_Time'] <= three_days_later) &
                        (out['Status'] == 'Shut-in')
                    ].index
                    
                    if len(shutin_indices) > 0:
                        # ambil Shut-in terjauh (notebook line 977)
                        last_shutin_idx = shutin_indices[-1]
                        logger.info(f"Found Shut-in, marking Start-up Phase from idx {first_after_gap_idx} to {last_shutin_idx}")
                        # semua selain Shut-in antara first_after_gap_idx sampai last_shutin_idx → Start-up Phase
                        # Notebook lines 979-982
                        for j in range(first_after_gap_idx, last_shutin_idx):
                            if out.loc[j, 'Status'] != 'Shut-in':
                                out.at[j, 'Prediction'] = 13
                                out.at[j, 'Status'] = 'Start-up Phase'
                                # Date column already added above
                    else:
                        # tidak ada Shut-in, ubah EDP 24 jam ke depan menjadi Start-up Phase
                        # Notebook lines 984-993
                        end_24h = first_after_gap_time + pd.Timedelta(hours=24)
                        edp_indices = out[
                            (out['Window_Start_Time'] >= first_after_gap_time) &
                            (out['Window_Start_Time'] <= end_24h) &
                            (out['Status'].str.contains('Electrical Downhole Problem', na=False))
                        ].index
                        logger.info(f"No Shut-in found, marking {len(edp_indices)} EDP as Start-up Phase")
                        for j in edp_indices:
                            out.at[j, 'Prediction'] = 13
                            out.at[j, 'Status'] = 'Start-up Phase'
                            # Date column already added above

        except Exception as e:
            logger.warning(f"Failed applying additional status rules: {e}")

        return out

    def _load_wc_data(self) -> None:
        """Load daily Watercut (production) data flexibly.
        Priority:
        1) self.prod_data_path (file) if provided
        2) self.prod_data_path (dir)/{well}.csv if dir provided
        3) Sibling of input: replace 'Data Sensor' with 'Data Produksi' and use {well}.csv
        4) Project 'Test Web/Data Produksi/{well}.csv'
        5) Project root 'prod_data.csv' (optionally filter by Well column if present)
        """
        project_root = Path(__file__).resolve().parents[1]
        attempted = []

        # Build candidate paths
        candidates: List[Path] = []

        # 1) Explicit path from user (file or directory)
        if getattr(self, 'prod_data_path', None):
            p = Path(self.prod_data_path)
            if p.suffix.lower() == '.csv':
                candidates.append(p)
            else:
                candidates.append(p / f'{self.well_name}.csv')

        # 2) Sibling of input file under 'Data Produksi'
        if getattr(self, 'input_file_path', None):
            try:
                inp = Path(self.input_file_path)
                # Try to locate repo root by finding 'Test Web' part
                parts = list(inp.parts)
                if 'Test Web' in parts:
                    idx = parts.index('Test Web')
                    repo_root = Path(*parts[:idx+1])
                else:
                    repo_root = project_root
                candidates.append(repo_root / 'Data Produksi' / f'{self.well_name}.csv')
                candidates.append(repo_root / 'Hasil Bacaan Notebook' / self.well_name / 'prod_data.csv')
            except Exception:
                pass

        # 3) Project standard location
        candidates.append(project_root / 'Test Web' / 'Data Produksi' / f'{self.well_name}.csv')
        # 4) Project root prod_data.csv
        candidates.append(project_root / 'prod_data.csv')

        csv_path_used: Optional[Path] = None
        df = None
        for cand in candidates:
            attempted.append(str(cand))
            try:
                if cand.exists():
                    # First try: normal read
                    df_try = pd.read_csv(cand)
                    
                    # Check if first data row looks like a units row (e.g., "psig", "%", "hours")
                    # If so, skip it by re-reading with skiprows=[1]
                    if len(df_try) > 0 and {'Date', 'WC'}.issubset(df_try.columns):
                        first_date = str(df_try['Date'].iloc[0]).lower()
                        first_wc = str(df_try['WC'].iloc[0]).lower()
                        # Detect unit row patterns
                        if any(unit in first_date for unit in ['psig', 'hours', 'bopd', 'bwpd', '"/64"']) or \
                           first_wc in ['%', 'scf/bbl', 'bbls', 'mbbls', 'bopd', 'bwpd']:
                            # Re-read skipping row 1 (the units row)
                            df_try = pd.read_csv(cand, skiprows=[1])
                    
                    # Validate minimal columns
                    if {'Date', 'WC'}.issubset(df_try.columns):
                        df = df_try
                        csv_path_used = cand
                        break
                    # Try wide format: per-well columns
                    if 'Date' in df_try.columns and self.well_name in df_try.columns:
                        df = df_try.rename(columns={self.well_name: 'WC'})[['Date', 'WC']]
                        csv_path_used = cand
                        break
                    # Try long format with 'Well' column
                    if {'Date', 'Well', 'WC'}.issubset(df_try.columns):
                        df = df_try[df_try['Well'] == self.well_name][['Date', 'WC']]
                        csv_path_used = cand
                        break
            except Exception:
                continue

        if df is None:
            raise FileNotFoundError(
                "Watercut data not found. Tried:\n - " + "\n - ".join(attempted)
            )

        # Parse dates with various formats, then normalize to date
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=False)
        df['Date'] = df['Date'].dt.normalize()

        # Clean WC strings like '100.00\xa0' or with commas/percent signs
        def to_num(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x)
            s = s.replace(',', '')
            s = ''.join(ch for ch in s if (ch.isdigit() or ch == '.' or ch == '-'))
            try:
                return float(s) if s not in ('', '-', '.') else np.nan
            except Exception:
                return np.nan

        df['WC'] = df['WC'].apply(to_num)
        self.df_wc = df[['Date', 'WC']].dropna(subset=['Date']).reset_index(drop=True)
        logger.info(f"Loaded Watercut data with {len(self.df_wc)} rows from {csv_path_used}")

    def _export_notebook_style_csv(self, filename: str, dataframe: pd.DataFrame) -> None:
        """Replicate notebook CSV exports while stripping internal helper columns."""
        try:
            project_root = Path(__file__).resolve().parents[1]
            export_path = project_root / filename
            export_df = dataframe.copy()
            internal_cols = [c for c in export_df.columns if c.startswith('predicted_')]
            if internal_cols:
                export_df = export_df.drop(columns=internal_cols, errors='ignore')
            export_df.to_csv(export_path, index=False)
            logger.info(f"Exported notebook-style CSV to: {export_path}")
        except Exception as exc:
            logger.warning(f"Failed to export {filename}: {exc}")
    
    def _save_results(self, results: Dict[str, np.ndarray]) -> None:
        """Save prediction results to CSV files.
        
        Args:
            results: Dictionary containing the prediction results
        """
        # Use configured output directory
        output_dir = str(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving results to directory: {output_dir}")
        
        for model_name, predictions in results.items():
            # The failure prediction has a dedicated saver; skip raw save here
            if model_name == 'failure_prediction':
                logger.info("Skipping raw save for failure_prediction (handled by _save_failure_results)")
                continue
            try:
                # Create output file with absolute path in home directory
                output_file = os.path.join(output_dir, f"{self.well_name}_{model_name}_predictions.csv")
                logger.info(f"Saving {model_name} predictions to: {output_file}")

                # Normalize predictions to numpy array
                pred_arr = np.asarray(predictions)

                # Build predictions-only DataFrame first
                if pred_arr.ndim == 1:
                    result_df = pd.DataFrame({'prediction': pred_arr})
                elif pred_arr.ndim == 2:
                    result_df = pd.DataFrame({f'prediction_{i}': pred_arr[:, i] for i in range(pred_arr.shape[1])})
                else:
                    result_df = pd.DataFrame({'prediction': pred_arr.reshape(-1)})

                # Optionally add timestamp if it matches row count
                if 'Reading Time' in self.data.columns:
                    ts = self.data['Reading Time']
                    if len(ts) == len(result_df):
                        result_df.insert(0, 'timestamp', ts)
                    else:
                        logger.warning("Timestamp length does not match predictions; saving without timestamp")

                # Handle empty predictions gracefully
                if result_df.shape[0] == 0:
                    logger.warning(f"No predictions to save for {model_name}; skipping file write")
                    continue

                # Save to CSV
                result_df.to_csv(
                    output_file,
                    index=False,
                    float_format='%.4f'
                )

                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    logger.info(f"Successfully saved {model_name} predictions to {output_file} (Size: {file_size} bytes)")

            except Exception as e:
                logger.error(f"Error saving {model_name} results: {str(e)}", exc_info=True)
                # Simpler fallback: predictions only
                try:
                    pd.DataFrame({'prediction': np.asarray(predictions).reshape(-1)}).to_csv(output_file, index=False)
                    logger.warning(f"Used fallback method to save {model_name} results (predictions only)")
                except Exception as inner_e:
                    logger.error(f"Failed to save results with fallback method: {str(inner_e)}")


    def _save_failure_results(self, final_df: pd.DataFrame) -> None:
        """Save the final failure prediction results in the template format
        with columns: Window_Start_Time, Prediction, Status, Recommendation.
        """
        output_dir = str(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{self.well_name}_failure_prediction_30min.csv")
        try:
            # Ensure ordering of columns (include Date column matching notebook for main file)
            base_cols = ['Window_Start_Time', 'Prediction', 'Status', 'Recommendation', 'Date']
            df_to_save = final_df[base_cols].copy() if all(c in final_df.columns for c in base_cols) else final_df.copy()
            if 'Recommendation' in df_to_save.columns:
                df_to_save['Recommendation'] = (
                    df_to_save['Recommendation']
                    .astype(str)
                    .str.replace('\n', ' ', regex=False)
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.strip()
                )
            df_to_save.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
            logger.info(f"Saved final failure results to: {output_file}")

            # Also save notebook-style prediction_results_30menit.csv at project root, well output, and stress_test
            try:
                project_root = Path(__file__).resolve().parents[1]
                df_nb = df_to_save.copy()
                if 'Recommendation' in df_nb.columns:
                    df_nb['Recommendation'] = df_nb['Recommendation'].astype(str).str.strip()
                # Notebook prediction_results_30menit.csv only keeps 30-min view:
                # Window_Start_Time, Status, Recommendation (no Prediction/Date)
                nb_cols = ['Window_Start_Time', 'Status', 'Recommendation']
                for c in nb_cols:
                    if c not in df_nb.columns:
                        df_nb[c] = ''
                df_nb = df_nb[nb_cols]

                # Write to project root
                pred30_path = project_root / 'prediction_results_30menit.csv'
                df_nb.to_csv(pred30_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
                logger.info(f"Saved notebook-style 30-minute predictions to: {pred30_path}")

                # Write to well output directory for comparator
                try:
                    pred30_out = Path(output_dir) / 'prediction_results_30menit.csv'
                    df_nb.to_csv(pred30_out, index=False, date_format='%Y-%m-%d %H:%M:%S')
                    logger.info(f"Saved notebook-style predictions to well output: {pred30_out}")
                except Exception as e_out:
                    logger.warning(f"Could not save prediction_results_30menit.csv to well output: {e_out}")

                # If stress_test directory exists, also write there for easy diffing
                stress_dir = project_root / 'stress_test'
                if stress_dir.exists():
                    pred30_path_stress = stress_dir / 'prediction_results_30menit.csv'
                    df_nb.to_csv(pred30_path_stress, index=False, date_format='%Y-%m-%d %H:%M:%S')
                    logger.info(f"Saved notebook-style predictions to stress_test: {pred30_path_stress}")
            except Exception as e2:
                logger.warning(f"Could not save prediction_results_30menit.csv: {e2}")
            
            # Generate 3-hour aggregated results (like notebook)
            try:
                self._save_3hour_aggregated_results(final_df)
            except Exception as e3:
                logger.warning(f"Could not save 3-hour aggregated results: {e3}")
                
        except Exception as e:
            logger.error(f"Error saving final failure results: {e}")

    def _save_3hour_aggregated_results(self, final_df: pd.DataFrame) -> None:
        """Aggregate 30-minute predictions to 3-hour windows and save as result_df_3 jam.csv
        using the exact notebook majority and tie-break rules, leveraging daily problem counts.
        """
        if final_df.empty:
            logger.warning("No data to aggregate to 3-hour windows")
            return

        # Prepare dataframe
        df = final_df.copy()
        df['Window_Start_Time'] = pd.to_datetime(df['Window_Start_Time'])
        df = df.sort_values('Window_Start_Time').reset_index(drop=True)

        # Daily non-Running problem counts (exclude Running and Shut-in)
        df_idx = df.set_index('Window_Start_Time')
        df_idx['Date'] = df_idx.index.date
        daily_problem_counts_dict = {}
        for date, group in df_idx.groupby('Date'):
            non_running = group[(group['Status'] != 'Running') & (group['Status'] != 'Shut-in')]
            daily_problem_counts_dict[date] = Counter(non_running['Status'])

        # Resample per 3 hours and compute dominant status
        grouped = df_idx.resample('3H')
        results = []
        for timestamp, group in grouped:
            total = len(group)
            if total == 0:
                continue

            # Majority Shut-in wins
            shutin_count = (group['Status'] == 'Shut-in').sum()
            if shutin_count > total / 2:
                dominant = 'Shut-in'
            else:
                # Exclude Shut-in
                group_no_shutin = group[group['Status'] != 'Shut-in']
                total_valid = len(group_no_shutin)
                if total_valid == 0:
                    continue  # all Shut-in handled above

                running_count = (group_no_shutin['Status'] == 'Running').sum()
                non_running_count = total_valid - running_count
                window_date = group.index[0].date()
                day_problem_counter = daily_problem_counts_dict.get(window_date, {})

                if non_running_count >= (total_valid / 2):
                    status_counts = group_no_shutin['Status'].value_counts()
                    status_counts_no_running = status_counts.drop('Running', errors='ignore')
                    if len(status_counts_no_running) == 0:
                        dominant = 'Running'
                    else:
                        top_count = status_counts_no_running.max()
                        top_statuses = status_counts_no_running[status_counts_no_running == top_count]
                        if len(top_statuses) == 1:
                            dominant = top_statuses.idxmax()
                        else:
                            tie_candidates = list(top_statuses.index)
                            tie_day_counts = {s: day_problem_counter.get(s, 0) for s in tie_candidates}
                            dominant = max(tie_day_counts, key=tie_day_counts.get) if tie_day_counts else tie_candidates[0]
                else:
                    dominant = 'Running'

            results.append({'Window_Start_Time': timestamp, 'Dominant Status': dominant})

        result_3h = pd.DataFrame(results)
        
        # Map Dominant Status ke recommendation
        def recommendation_by_status(status: str) -> str:
            """Map status ke recommendation text."""
            status_to_pred = {
                'Running': 0,
                'Low PI': 1,
                'Pump Wear': 2,
                'Tubing Leak': 3,
                'Higher PI': 4,
                'Increase in Frequency': 5,
                'Open Choke': 6,
                'Increase in Watercut': 7,
                'Sand Ingestion': 8,
                'Closed Valve': 9,
                'Electrical Downhole Problem': 10,
                'Shut-in': 11,
                '100% Watercut': 12,
                'Start-up Phase': 13,
            }
            pred_num = status_to_pred.get(status, 0)
            
            # Use existing recommendation_map function from _assemble_failure_results
            recs = {
                0: " ",
                1: (
                    "The Possibility Causes: 1. Well productivity less than pump design range 2. Restricted pump. "
                    "1. Analyze the fluid level and Bottom Hole Pressure (BHP) data! If in acceptable range, adjust the tubing well head "
                    "pressure and bring the pump production rate within design rate. 2. Check the possibility of restricted pump! "
                    "Pumping fluids through tubing when water sources are available."
                ),
                2: (
                    "1. Check performance drop (>15–20% from initial installation) "
                    "2. Verify vibration increase (>20%) 3. Perform shut-in test with surface check valve "
                    "closed while pump is running"
                ),
                3: (
                    "1. Confirm by a pressure test at the tubing wellhead "
                    "2. Meanwhile, fill up the tubing and pressure up against RCV"
                ),
                4: (
                    "The Possibility Causes: Well productivity above pump design range "
                    "1. Analyze fluid level and BHP 2. Adjust wellhead pressure to maintain design rate "
                    "The Possibility Causes: Change in fluid characteristics 1. Analyze fluid level and BHP "
                    "2. Conduct fluid analysis for pump re-design reference"
                ),
                5: (
                    "1. Compare discharge pressure with historical data "
                    "2. Reduce frequency via VSD"
                ),
                6: (
                    "Check pump discharge pressure and production rate, "
                    "compare with historical well data"
                ),
                7: (
                    "Adjust tubing wellhead pressure to bring production rate "
                    "within design limits"
                ),
                8: (
                    "1. Check flow line and separator for evidence of sand, mud, "
                    "or debris 2. Design solid control system for next installation"
                ),
                9: (
                    "1. Verify if the valve was deliberately partially closed by "
                    "Field Service Tech 2. Contact the Field Technician for on-site inspection"
                ),
                10: (
                    "1. Verify surface equipment (VSD, transformer, junction box) "
                    "to isolate downhole issue 2. Perform VSD soft shutdown to prevent reverse current damage "
                    "3. Conduct a DIFA (Dismantle Inspection and Failure Analysis"
                ),
                11: (
                    "Shut-in detected. Verify operating schedule and surface conditions. Ensure Amps/Frequency "
                    "are expected to be zero."
                ),
                12: (
                    "The Possibility Causes: Well producing 100% water—possible water "
                    "breakthrough or reservoir depletion 1. Check production test and GOR trend to confirm water "
                    "source 2. Inspect well completion for leaks; consider shut-in, isolation, or water-shutoff treatment"
                ),
                13: (
                    "Start-up Phase after extended data gap: 1. Conduct no-load test before commissioning to ensure "
                    "proper drive operation 2. Verify drive parameters, transformer taps, and gauge units are correctly "
                    "set 3. Test both rotations; select the one with lower amperage (correct pump rotation) "
                    "4. Monitor WHP—if no buildup, briefly choke the well to initiate flow and confirm ΔP"
                ),
            }
            return recs.get(pred_num, " ")
        
        result_3h['Recommendation'] = result_3h['Dominant Status'].apply(recommendation_by_status)

        # Match notebook timestamp formatting (ISO vs M/D/YYYY H:MM)
        desired_iso = False
        try:
            project_root = Path(__file__).resolve().parents[1]
            nb_file = project_root / 'Test Web' / 'Hasil Bacaan Notebook' / self.well_name / 'result_df_3 jam.csv'
            if nb_file.exists():
                with open(nb_file, 'r', encoding='utf-8', errors='ignore') as f:
                    _ = f.readline()
                    sample = f.readline().strip().split(',')[0]
                    if '-' in sample:  # e.g., 2025-02-01 09:00:00
                        desired_iso = True
        except Exception:
            pass

        try:
            result_3h['Window_Start_Time'] = pd.to_datetime(result_3h['Window_Start_Time'])
            if desired_iso:
                formatted = result_3h['Window_Start_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                formatted = result_3h['Window_Start_Time'].dt.strftime('%-m/%-d/%Y %-H:%M')
        except Exception:
            if desired_iso:
                formatted = result_3h['Window_Start_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                formatted = result_3h['Window_Start_Time'].dt.strftime('%m/%d/%Y %H:%M')
                formatted = formatted.str.lstrip('0').str.replace('/0', '/', regex=False)
                formatted = formatted.str.replace(' 0([0-9]):', r' \1:', regex=True)
        result_3h['Window_Start_Time'] = formatted
        
        # Save results (well folder + project root)
        output_dir = str(self.output_dir)
        output_file = os.path.join(output_dir, "result_df_3 jam.csv")
        result_3h.to_csv(output_file, index=False)
        logger.info(f"Saved 3-hour aggregated results to: {output_file}")
        try:
            project_root = Path(__file__).resolve().parents[1]
            root_file = project_root / 'result_df_3 jam.csv'
            result_3h.to_csv(root_file, index=False)
            logger.info(f"Saved 3-hour aggregated results to project root: {root_file}")
        except Exception as e:
            logger.warning(f"Could not save 3-hour results to project root: {e}")
    
    def _slope_symbol(self, value: float) -> str:
        """Tentukan simbol arah berdasarkan nilai slope."""
        if value >= 0.005:
            return "↑"
        elif value <= 0.005:
            return "↓"
        else:
            return "→"
    
    def _make_indicator(self, row: pd.Series) -> str:
        """Buat kolom indicator berdasarkan status dan slopes."""
        # Mapping kolom slope per status
        status_slope_map = {
            "Low PI": ["A", "IP", "DP", "R"],
            "Pump Wear": ["A", "IP", "DP", "V", "R"],
            "Tubing Leak": ["A", "IP", "DP", "IT", "MT", "R"],
            "Higher PI": ["A", "IP", "DP", "R"],
            "Increase in Frequency": ["A", "IP", "DP", "MT", "R"],
            "Open Choke": ["A", "IP", "DP", "MT", "R"],
            "Increase in Watercut": ["A", "IP", "DP", "MT", "R"],
            "Sand Ingestion": ["A", "IP", "DP", "MT", "V", "R"],
            "Closed Valve": ["A", "IP", "DP", "IT", "MT", "R"]
        }
        
        status = row["Status"]
        
        # Kosong untuk status tertentu
        if status in ["Running", "Shut-in", "Start-up Phase"]:
            return ""
        
        # Custom text untuk status khusus
        if status == "100% Watercut":
            return "100% WC in Prod"
        elif status == "Electrical Downhole Problem":
            return "A and Freq 0, others constant"
        
        # Gunakan daftar kolom sesuai mapping
        cols = status_slope_map.get(status, [])
        if not cols:
            return ""
        
        indicators = []
        for col in cols:
            if col in row.index and pd.notna(row[col]):
                indicators.append(f"{col}{self._slope_symbol(row[col])}")
        
        return " ".join(indicators)
    
    def _combine_indicators(self, indicators: List[str]) -> str:
        """Gabungkan indikator dari beberapa window dengan simbol yang digabungkan."""
        symbol_dict = {}
        
        for ind in indicators:
            if not ind or not isinstance(ind, str):
                continue
            pairs = re.findall(r"([A-Z]+)([↑↓→])", ind)
            for var, sym in pairs:
                symbol_dict.setdefault(var, set()).add(sym)
        
        # Urutan kolom tetap mengikuti urutan logis
        col_order = ["A", "IP", "DP", "IT", "MT", "V", "R"]
        
        combined = []
        for col in col_order:
            if col in symbol_dict:
                # Urutkan simbol agar konsisten
                sorted_syms = "".join(sorted(symbol_dict[col], key=lambda x: "→↑↓".index(x)))
                combined.append(f"{col}{sorted_syms}")
        
        return " ".join(combined)
    
    def _add_indicators_to_results(self, indicator_df: pd.DataFrame, result_3jam: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Tambahkan kolom Indicator ke hasil 30 menit dan 3 jam."""
        # 1. Tambahkan Indicator ke data 30 menit
        indicator = indicator_df.copy()
        
        # Buat kolom indicator dengan mapping
        indicator["Indicator"] = indicator.apply(self._make_indicator, axis=1)

        # Susun dataframe 30-menit agar persis seperti notebook:
        # Window_Start_Time, Indicator, Status, Recommendation
        base_cols = ["Window_Start_Time", "Indicator", "Status", "Recommendation"]
        for c in base_cols:
            if c not in indicator.columns:
                indicator[c] = ""
        indicator = indicator[base_cols]
        
        # 2. Tambahkan Indicator ke data 3 jam
        result_3h = result_3jam.copy()
        indicators_combined = []
        
        for _, row in result_3h.iterrows():
            start_time = pd.to_datetime(row["Window_Start_Time"])
            end_time = start_time + timedelta(hours=3)
            status = row["Dominant Status"]
            
            # Tangani status khusus langsung
            if status == "100% Watercut":
                indicators_combined.append("100% WC in Prod")
                continue
            elif status == "Electrical Downhole Problem":
                indicators_combined.append("A and Freq 0, others constant")
                continue
            elif status in ["Running", "Shut-in", "Start-up Phase"]:
                indicators_combined.append("")
                continue
            
            # Untuk status lainnya, ambil subset 3 jam
            subset = indicator[
                (pd.to_datetime(indicator["Window_Start_Time"]) >= start_time)
                & (pd.to_datetime(indicator["Window_Start_Time"]) < end_time)
                & (indicator["Status"] == status)
            ]
            
            combined = self._combine_indicators(subset["Indicator"].tolist())
            indicators_combined.append(combined)
        
        result_3h["Indicator"] = indicators_combined
        
        # Susun ulang kolom: Window_Start_Time, Indicator, Dominant Status, Recommendation
        cols_3h_order = ["Window_Start_Time", "Indicator", "Dominant Status"]
        for col in result_3h.columns:
            if col not in cols_3h_order:
                cols_3h_order.append(col)
        result_3h = result_3h[[c for c in cols_3h_order if c in result_3h.columns]]
        
        # Urutkan berdasarkan waktu
        result_3h = result_3h.sort_values("Window_Start_Time", ascending=True).reset_index(drop=True)
        indicator = indicator.sort_values("Window_Start_Time", ascending=True).reset_index(drop=True)
        
        return indicator, result_3h
    
    def _parse_indicator(self, ind_text: str) -> List[str]:
        """Parse indicator text menjadi list teks yang mudah dibaca."""
        param_map = {
            "A": "Ampere",
            "IP": "Intake Pressure",
            "DP": "Discharge Pressure",
            "IT": "Intake Temperature",
            "MT": "Motor Temperature",
            "V": "Vibration",
            "Q": "Rate"
        }
        
        def arrow_meaning(symbol):
            return {
                "↑": "Increase",
                "↓": "Decrease",
                "→": "Stable"
            }.get(symbol, "")
        
        if not isinstance(ind_text, str) or not ind_text.strip():
            return []
        
        pairs = re.findall(r"([A-Z]+)([↑↓→]+)", ind_text)
        result = []
        for var, syms in pairs:
            if var in param_map:
                for s in syms:
                    result.append(f"{param_map[var]} ({s}): {arrow_meaning(s)}")
        return result
    
    def _wrap_text(self, text: str, width: int = 38) -> str:
        """Wrap long text lines so they fit nicely inside the table."""
        if not isinstance(text, str) or text.strip() == "":
            return ""
        return "\n".join(textwrap.wrap(text, width=width))
    
    def _print_report(self, title: str, failure_row: Optional[pd.Series], status_row: Optional[pd.Series]) -> str:
        """Print a side-by-side failure vs status report in a table-like format."""
        output = []
        output.append(f"\n# {title}")
        output.append("-" * 76)
        
        # Set column widths
        col_width = 38
        divider = " | "
        
        # Extract key data
        fail_time = failure_row["Window_Start_Time"] if failure_row is not None else "-"
        stat_time = status_row["Window_Start_Time"] if status_row is not None else "-"
        
        # Get status/indicator/recommendation
        if "Dominant Status" in failure_row.index if failure_row is not None else False:
            fail_status = failure_row["Dominant Status"]
            fail_ind = failure_row.get("Indicator", "")
            fail_rec = failure_row.get("Recommendation", "")
        else:
            fail_status = failure_row["Status"] if failure_row is not None else "-"
            fail_ind = failure_row.get("Indicator", "") if failure_row is not None else ""
            fail_rec = failure_row.get("Recommendation", "") if failure_row is not None else ""
        
        if "Dominant Status" in status_row.index if status_row is not None else False:
            stat_status = status_row["Dominant Status"]
            stat_ind = status_row.get("Indicator", "")
            stat_rec = status_row.get("Recommendation", "")
        else:
            stat_status = status_row["Status"] if status_row is not None else "-"
            stat_ind = status_row.get("Indicator", "") if status_row is not None else ""
            stat_rec = status_row.get("Recommendation", "") if status_row is not None else ""
        
        # Header row
        output.append(f"{'Latest Failure'.ljust(col_width)}{divider}{'Latest Status'.ljust(col_width)}")
        
        # Time and status rows (bold dengan **)
        output.append(f"**{fail_time}**".ljust(col_width + 4) + divider + f"**{stat_time}**")
        output.append(f"**{fail_status}**".ljust(col_width + 4) + divider + f"**{stat_status}**")
        
        # Indicators
        indicators_fail = self._parse_indicator(fail_ind)
        indicators_stat = self._parse_indicator(stat_ind)
        
        output.append(f"{'Indicator:'.ljust(col_width)}{divider}{'Indicator:'.ljust(col_width)}")
        
        # Pad untuk memastikan jumlah baris sama
        max_ind_len = max(len(indicators_fail), len(indicators_stat))
        for i in range(max_ind_len):
            f_line = indicators_fail[i] if i < len(indicators_fail) else ""
            s_line = indicators_stat[i] if i < len(indicators_stat) else ""
            output.append(f"{f_line.ljust(col_width)}{divider}{s_line.ljust(col_width)}")
        
        # Recommendations section
        fail_reco_wrapped = self._wrap_text(str(fail_rec), width=col_width - 2)
        stat_reco_wrapped = self._wrap_text(str(stat_rec), width=col_width - 2)
        
        output.append(f"{'Recommendation:'.ljust(col_width)}{divider}{'Recommendation:'.ljust(col_width)}")
        
        fail_lines = fail_reco_wrapped.split("\n") if fail_reco_wrapped else [""]
        stat_lines = stat_reco_wrapped.split("\n") if stat_reco_wrapped else [""]
        max_len = max(len(fail_lines), len(stat_lines))
        
        for i in range(max_len):
            fail_line = fail_lines[i] if i < len(fail_lines) else ""
            stat_line = stat_lines[i] if i < len(stat_lines) else ""
            output.append(f"{fail_line.ljust(col_width)}{divider}{stat_line.ljust(col_width)}")
        
        output.append("-" * 76)
        
        return "\n".join(output)
    
    def _generate_latest_report(self, indicator_30min: pd.DataFrame, result_3h: pd.DataFrame) -> Dict:
        """Generate latest status and latest failure data untuk Flask UI."""
        ignore_status = ["Running", "Shut-in", "Start-up Phase"]
        
        # Pastikan kolom waktu sudah datetime
        indicator_30min = indicator_30min.copy()
        result_3h = result_3h.copy()
        indicator_30min["Window_Start_Time"] = pd.to_datetime(indicator_30min["Window_Start_Time"])
        result_3h["Window_Start_Time"] = pd.to_datetime(result_3h["Window_Start_Time"])
        
        report_data = {
            "3_hours": {},
            "30_minutes": {}
        }
        
        # ========== 3 HOURS READING ==========
        # Latest Status (termasuk Running)
        latest_status_3h = (
            result_3h
            .sort_values("Window_Start_Time", ascending=False)
            .head(1)
        )
        
        # Latest Failure (exclude normal status)
        latest_failure_3h = (
            result_3h[~result_3h["Dominant Status"].isin(ignore_status)]
            .sort_values("Window_Start_Time", ascending=False)
            .head(1)
        )
        
        if not latest_status_3h.empty:
            stat_row = latest_status_3h.iloc[0]
            report_data["3_hours"]["latest_status"] = {
                "time": str(stat_row["Window_Start_Time"]),
                "status": stat_row.get("Dominant Status", "-"),
                "indicator": stat_row.get("Indicator", ""),
                "indicator_parsed": self._parse_indicator(stat_row.get("Indicator", "")),
                "recommendation": stat_row.get("Recommendation", "")
            }
        
        if not latest_failure_3h.empty:
            fail_row = latest_failure_3h.iloc[0]
            report_data["3_hours"]["latest_failure"] = {
                "time": str(fail_row["Window_Start_Time"]),
                "status": fail_row.get("Dominant Status", "-"),
                "indicator": fail_row.get("Indicator", ""),
                "indicator_parsed": self._parse_indicator(fail_row.get("Indicator", "")),
                "recommendation": fail_row.get("Recommendation", "")
            }
        
        # ========== 30 MINUTES READING ==========
        # Latest Status (termasuk Running)
        latest_status_30 = (
            indicator_30min
            .sort_values("Window_Start_Time", ascending=False)
            .head(1)
        )
        
        # Latest Failure (exclude normal status)
        latest_failure_30 = (
            indicator_30min[~indicator_30min["Status"].isin(ignore_status)]
            .sort_values("Window_Start_Time", ascending=False)
            .head(1)
        )
        
        if not latest_status_30.empty:
            stat_row = latest_status_30.iloc[0]
            report_data["30_minutes"]["latest_status"] = {
                "time": str(stat_row["Window_Start_Time"]),
                "status": stat_row.get("Status", "-"),
                "indicator": stat_row.get("Indicator", ""),
                "indicator_parsed": self._parse_indicator(stat_row.get("Indicator", "")),
                "recommendation": stat_row.get("Recommendation", "")
            }
        
        if not latest_failure_30.empty:
            fail_row = latest_failure_30.iloc[0]
            report_data["30_minutes"]["latest_failure"] = {
                "time": str(fail_row["Window_Start_Time"]),
                "status": fail_row.get("Status", "-"),
                "indicator": fail_row.get("Indicator", ""),
                "indicator_parsed": self._parse_indicator(fail_row.get("Indicator", "")),
                "recommendation": fail_row.get("Recommendation", "")
            }
        
        return report_data
    
    def _open_in_os(self, path: Path) -> None:
        """Try to open a file in the OS default viewer (macOS, Windows, Linux)."""
        try:
            p = str(path)
            if sys.platform.startswith('darwin'):
                subprocess.Popen(['open', p])
            elif sys.platform.startswith('win'):
                os.startfile(p)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(['xdg-open', p])
        except Exception as e:
            logger.debug(f"Could not auto-open {path}: {e}")

    def plot_results(self, results: Dict[str, np.ndarray]) -> None:
        """Generate and display time series plots:
        - Discharge Pressure (actual vs predicted)
        - Virtual Rate (actual vs predicted)
        - Sensor overview with normalized signals; bottom panel overlays predicted DP and VR

        Saves PNGs to data/output and attempts to open them in the OS viewer.
        """
        try:
            if self.data is None or len(self.data) == 0:
                logger.warning("No data available to plot.")
                return

            df = self.data.copy()
            time_col = 'Reading Time' if 'Reading Time' in df.columns else None
            if time_col:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                df = df.dropna(subset=[time_col]).sort_values(time_col)

            # Attach predictions if present and length matches
            if 'discharge_pressure' in results and len(results['discharge_pressure']) == len(df):
                df['predicted_discharge_pressure'] = results['discharge_pressure']
            if 'virtual_rate' in results and len(results['virtual_rate']) == len(df):
                df['predicted_virtual_rate'] = results['virtual_rate']

            self.output_dir.mkdir(parents=True, exist_ok=True)
            sns.set(style='whitegrid')

            # X-axis
            x = df[time_col] if time_col else np.arange(len(df))

            # Figure 1: Discharge Pressure
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            if 'Discharge Pressure (psi) (Raw)' in df.columns and df['Discharge Pressure (psi) (Raw)'].notna().any():
                ax1.plot(x, df['Discharge Pressure (psi) (Raw)'], label='DP actual', color='#4C78A8', linewidth=1)
            if 'predicted_discharge_pressure' in df.columns:
                ax1.plot(x, df['predicted_discharge_pressure'], label='DP predicted', color='#F58518', linewidth=1)
            ax1.set_title(f"{self.well_name} - Discharge Pressure")
            ax1.set_xlabel('Time' if time_col else 'Index')
            ax1.set_ylabel('psi')
            ax1.legend()
            fig1.tight_layout()
            dp_plot = self.output_dir / f"{self.well_name}_discharge_pressure_plot.png"
            fig1.savefig(dp_plot, dpi=150)

            # Figure 2: Virtual Rate
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            if 'Virtual Rate (BFPD) (Raw)' in df.columns and df['Virtual Rate (BFPD) (Raw)'].notna().any():
                ax2.plot(x, df['Virtual Rate (BFPD) (Raw)'], label='VR actual', color='#4C78A8', linewidth=1)
            if 'predicted_virtual_rate' in df.columns:
                ax2.plot(x, df['predicted_virtual_rate'], label='VR predicted', color='#F58518', linewidth=1)
            ax2.set_title(f"{self.well_name} - Virtual Rate")
            ax2.set_xlabel('Time' if time_col else 'Index')
            ax2.set_ylabel('BFPD')
            ax2.legend()
            fig2.tight_layout()
            vr_plot = self.output_dir / f"{self.well_name}_virtual_rate_plot.png"
            fig2.savefig(vr_plot, dpi=150)

            # Figure 3: Sensor overview + predictions
            fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            sensor_cols = [
                'Average Amps (A) (Raw)',
                'Drive Frequency (Hz) (Raw)',
                'Intake Pressure (psi) (Raw)',
                'Motor Temperature (F) (Raw)',
                'Vibration (gravit) (Raw)'
            ]
            colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2']
            plotted_any = False
            for col, c in zip(sensor_cols, colors):
                if col in df.columns and df[col].notna().any():
                    series = df[col].astype(float)
                    # Z-score normalize to make scales comparable
                    mean = series.mean()
                    std = series.std() if series.std() else 1.0
                    ax3.plot(x, (series - mean) / std, label=col, color=c, linewidth=0.9)
                    plotted_any = True
            if plotted_any:
                ax3.set_title('Sensor overview (z-score normalized)')
                ax3.set_ylabel('z-score')
                ax3.legend(ncol=3, fontsize=8)

            # Bottom: predictions overlay
            if 'predicted_discharge_pressure' in df.columns:
                ax4.plot(x, df['predicted_discharge_pressure'], label='DP predicted', color='#F58518', linewidth=1)
            if 'predicted_virtual_rate' in df.columns:
                ax4.plot(x, df['predicted_virtual_rate'], label='VR predicted', color='#4C78A8', linewidth=1)
            ax4.set_title('Predictions')
            ax4.set_xlabel('Time' if time_col else 'Index')
            ax4.legend()
            fig3.tight_layout()
            overview_plot = self.output_dir / f"{self.well_name}_overview_plot.png"
            fig3.savefig(overview_plot, dpi=150)

            # Close all figures to prevent display
            plt.close('all')

            logger.info(f"Saved plots to: {dp_plot}, {vr_plot}, {overview_plot}")
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")

def main():
    """Example usage of the WellAnalysisPipeline."""
    # Example well name (update this with your actual well name)
    well_name = "SKW-02"
    
    # Initialize the pipeline
    pipeline = WellAnalysisPipeline(well_name)
    
    try:
        # Run the full analysis
        results = pipeline.run_full_analysis()
        
        # Generate and save plots
        pipeline.plot_results(results)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
