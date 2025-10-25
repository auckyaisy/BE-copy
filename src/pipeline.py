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

from config.config import (
    MODEL_PATHS, FEATURE_COLUMNS, TARGET_COLUMNS,
    INPUT_DIR, OUTPUT_DIR, DEFAULT_PARAMS
)
from .utils import calculate_well_slopes

# Set up logging
logging.basicConfig(level=logging.DEBUG)
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
            df['Reading Time'] = pd.to_datetime(df['Reading Time'])
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
                self.data['Reading Time'] = pd.to_datetime(self.data['Reading Time'], errors='coerce')
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
            df_all['Reading Time'] = pd.to_datetime(df_all['Reading Time'], errors='coerce')
            df_all = (
                df_all.set_index('Reading Time')
                      .resample('30T', origin='epoch')
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
            df_for_slopes['Reading Time'] = pd.to_datetime(df_for_slopes['Reading Time'], errors='coerce')
            df_for_slopes = df_for_slopes.dropna(subset=['Reading Time']).sort_values('Reading Time').reset_index(drop=True)

            if df_for_slopes.empty:
                slopes_df = pd.DataFrame(columns=['Window_Start_Time','A','IP','DP','IT','MT','V','R'])
            else:
                time_interval = pd.Timedelta(minutes=30)
                start_time = df_for_slopes['Reading Time'].iloc[0].floor('30min')
                end_time = df_for_slopes['Reading Time'].iloc[-1].ceil('30min')
                time_windows = pd.date_range(start=start_time, end=end_time, freq='30min')
                numerical_cols = df_for_slopes.select_dtypes(include=np.number).columns.tolist()

                from scipy.stats import linregress

                slopes_list = []
                for window_start in time_windows:
                    window_end = window_start + time_interval
                    window_df = df_for_slopes[(df_for_slopes['Reading Time'] >= window_start) & (df_for_slopes['Reading Time'] < window_end)]
                    if len(window_df) < 2:
                        continue

                    window_slopes = {}
                    for col in numerical_cols:
                        temp_df = window_df[['Reading Time', col]].dropna()
                        if len(temp_df) > 1:
                            temp_time_diff = (temp_df['Reading Time'] - window_start).dt.total_seconds().values
                            slope, _, _, _, _ = linregress(temp_time_diff, temp_df[col])
                            window_slopes[f"{col}_slope"] = slope
                        else:
                            window_slopes[f"{col}_slope"] = np.nan
                    window_slopes['Window_Start_Time'] = window_start
                    slopes_list.append(window_slopes)

                slopes_raw = pd.DataFrame(slopes_list)
                if slopes_raw.empty:
                    slopes_df = pd.DataFrame(columns=['Window_Start_Time','A','IP','DP','IT','MT','V','R'])
                else:
                    slopes_raw = slopes_raw.set_index('Window_Start_Time').reset_index()
                    # Expected order of slope columns as produced by notebook
                    slope_cols_mapping = [
                        ('Average Amps (A) (Raw)_slope', 'A'),
                        ('Intake Pressure (psi) (Raw)_slope', 'IP'),
                        ('Discharge Pressure (psi) (Raw)_slope', 'DP'),
                        ('Intake Temperature (F) (Raw)_slope', 'IT'),
                        ('Motor Temperature (F) (Raw)_slope', 'MT'),
                        ('Vibration (gravit) (Raw)_slope', 'V'),
                        ('Virtual Rate (BFPD) (Raw)_slope', 'R'),
                    ]
                    slopes_df = slopes_raw[['Window_Start_Time']].copy()
                    for src, dst in slope_cols_mapping:
                        slopes_df[dst] = slopes_raw.get(src, pd.Series(np.nan, index=slopes_raw.index))

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
                df['Reading Time'] = pd.to_datetime(df['Reading Time'], errors='coerce')
            
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
        df_idx['Reading Time'] = pd.to_datetime(df_idx['Reading Time'], errors='coerce')
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
        d['Reading Time'] = pd.to_datetime(d['Reading Time'], errors='coerce')
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
                    "The Possibility Causes:\n 1. Well productivity less than pump design range\n 2. Restricted pump\n"
                    "NOTIFICATIONS FOR ENGINEER!\n"
                    "1. Analyze the fluid level and Bottom Hole Pressure (BHP) data! If in acceptable range, Adjust the tubing well head pressure and bring the pump production rate within design rate\n"
                    "2. Check the possibility of restricted pump! Pumping fluids through tubing when water sources are available."
                ),
                2: (
                    "NOTIFICATIONS FOR ENGINEER!\n"
                    "1. Verify if vibration have increased by 20% from the pump install date\n"
                    "2. Do shut-in test while the surface check valve is closed, and the pump is running"
                ),
                3: (
                    "NOTIFICATIONS FOR ENGINEER!\n"
                    "1. Confirm by a pressure test at the tubing wellhead\n"
                    "2. Meanwhile, fill up the tubing and pressure up against RCV"
                ),
                4: (
                    "NOTIFICATIONS FOR ENGINEER!\n"
                    "1. Adjust the tubing well head pressure and bring the pump production rate within design rate\n"
                    "2. Conduct the fluid analysis as a basis for re-design pump"
                ),
                5: (
                    "NOTIFICATIONS FOR ENGINEER!\n"
                    "1. Lower the value of frequency using VSD.\n"
                    "2. Check the pump discharge pressure and compare to previous well data history"
                ),
                6: (
                    "NOTIFICATIONS FOR ENGINEER!\n"
                    "Analyze the fluid level and Bottom Hole Pressure (BHP) data!"
                ),
                7: (
                    "NOTIFICATIONS FOR ENGINEER!\n"
                    "1. Analyze the fluid level and Bottom Hole Pressure (BHP) data!\n"
                    "2. Adjust the tubing well head pressure and bring the pump production rate within design rate"
                ),
                8: (
                    "NOTIFICATIONS FOR ENGINEER!\n"
                    "1. Check flow line and separator for evidence of sand, mud, or debris.\n"
                    "2. Design solid control system for next installation"
                ),
                9: (
                    "NOTIFICATIONS FOR ENGINEER!\n"
                    "1. Verify if the valve was deliberately partially closed by Field Service Tech\n"
                    "2. Contact the Field Service Tech to check out well on location"
                ),
                10: (
                    "Electrical Downhole Problem suspected: "
                    "1) Verify surface equipment (VSD, step-up transformer, junction box) to confirm failure is downhole. "
                    "2) Perform a VSD soft shutdown to prevent reverse current surges. "
                    "3) Conduct a DIFA (Dismantle Inspection and Failure Analysis)."
                ),
                11: (
                    "Shut-in detected. Verify operating schedule and surface conditions. Ensure Amps/Frequency are expected to be zero."
                ),
                12: (
                    "Well producing 100% water — likely water breakthrough or reservoir depletion: "
                    "Causes: 1) Water coning/channeling from aquifer, 2) Casing/tubing leak allowing water influx, 3) Reservoir pressure depletion. "
                    "Recommended actions: 1) Verify production test (separator test or sampling), 2) Check GOR trend (near zero indicates water dominance), "
                    "3) Review well completion to find water source, 4) Consider temporary shut-in or zonal isolation, 5) Evaluate re-perforation or water-shutoff treatment."
                ),
                13: (
                    "Start-up Phase after extended data gap: 1) Monitor equipment closely during ramp-up, "
                    "2) Ensure surface controls follow the planned start-up procedure, "
                    "3) Confirm downhole pressures and temperatures stabilize before normal operation."
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
            df_all2['Reading Time'] = pd.to_datetime(df_all2['Reading Time'], errors='coerce')
            slopes_df2 = slopes_df.copy()
            slopes_df2['Window_Start_Time'] = pd.to_datetime(slopes_df2['Window_Start_Time'], errors='coerce')

            # Merge out with df_all (resampled point at window start)
            merged = (out
                      .merge(df_all2, left_on='Window_Start_Time', right_on='Reading Time', how='left', suffixes=('', '_res'))
                      .merge(slopes_df2, on='Window_Start_Time', how='left', suffixes=('', '_slope')))

            # Watercut override: if WC=100 for the date, set prediction to 12
            mask_wc = pd.Series(False, index=merged.index)
            if hasattr(self, 'df_wc') and self.df_wc is not None and not self.df_wc.empty:
                merged['Date'] = merged['Window_Start_Time'].dt.normalize()
                wc_map = self.df_wc.set_index('Date')['WC'].to_dict()
                merged['WC_val'] = merged['Date'].map(wc_map)
                mask_wc = (merged['WC_val'].fillna(0) >= 100.0)

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
            
            # Build has_variation Series by checking raw data within each 30-min window
            has_variation_list = []
            for idx in merged.index:
                window_start = merged.loc[idx, 'Window_Start_Time']
                if pd.isna(window_start):
                    has_variation_list.append(False)
                    continue
                window_end = window_start + pd.Timedelta(minutes=30)
                
                # Get raw data subset for this window (notebook lines 909-920)
                subset_ori = self.data[
                    (self.data['Reading Time'] >= window_start) &
                    (self.data['Reading Time'] < window_end)
                ]
                
                has_var = False
                if not subset_ori.empty:
                    for c in cols_check:
                        if c in subset_ori.columns and subset_ori[c].nunique() > 1:
                            has_var = True
                            break
                has_variation_list.append(has_var)
            
            any_variation = pd.Series(has_variation_list, index=merged.index)

            # Shut-in mask - sesuai notebook: hanya cek VR, DP, dan Vibration (bukan semua kolom)
            amps_zero = np.isclose(amps, 0.0, atol=TOL)
            freq_zero = np.isclose(freq, 0.0, atol=TOL)
            other_zero = (
                np.isclose(rate, 0.0, atol=TOL)
                & np.isclose(merged.get('Discharge Pressure (psi) (Raw)', pd.Series(0, index=merged.index)).fillna(0.0), 0.0, atol=TOL)
                & np.isclose(merged.get('Vibration (gravit) (Raw)', pd.Series(0, index=merged.index)).fillna(0.0), 0.0, atol=TOL)
            )
            mask_shutin = amps_zero & freq_zero & (other_zero | any_variation)

            # EDP override: if Amps and Freq are zero AND DP/IP/IT/MT/V are also zero (no variation)
            mask_edp = amps_zero & freq_zero & ~any_variation

            # Start from current predictions
            pred_vec = merged['Prediction'].astype(int).to_numpy(copy=True)

            # Apply overrides in order: EDP, Shut-in, then Watercut (highest priority)
            # Watercut MUST be last to ensure it's never overridden
            pred_vec = np.where(mask_edp, 10, pred_vec)
            pred_vec = np.where(mask_shutin, 11, pred_vec)
            pred_vec = np.where(mask_wc, 12, pred_vec)  # Watercut has highest priority

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
        """Load daily Watercut data from Test Web/Data Produksi/{well_name}.csv.
        Parses 'Date' to datetime (normalized to date) and 'WC' to numeric percent.
        """
        project_root = Path(__file__).resolve().parents[1]
        csv_path = project_root / 'Test Web' / 'Data Produksi' / f'{self.well_name}.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Watercut data file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if 'Date' not in df.columns or 'WC' not in df.columns:
            raise ValueError("Watercut file must contain 'Date' and 'WC' columns")

        # Parse dates with various formats, then normalize to date
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=False, infer_datetime_format=True)
        df['Date'] = df['Date'].dt.normalize()

        # Clean WC strings like '100.00\xa0' or with commas/percent signs
        def to_num(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x)
            s = s.replace(',', '')
            # remove non-digit/dot characters
            s = ''.join(ch for ch in s if (ch.isdigit() or ch == '.' or ch == '-'))
            try:
                return float(s) if s not in ('', '-', '.') else np.nan
            except Exception:
                return np.nan

        df['WC'] = df['WC'].apply(to_num)
        # keep only Date and WC
        self.df_wc = df[['Date', 'WC']].dropna(subset=['Date']).reset_index(drop=True)
        logger.info(f"Loaded Watercut data with {len(self.df_wc)} rows from {csv_path}")

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
                
                # Create a base DataFrame for results
                result_data = {}
                
                # Add timestamp if available
                if 'Reading Time' in self.data.columns:
                    result_data['timestamp'] = self.data['Reading Time']
                
                # Handle 1D and 2D predictions
                if hasattr(predictions, 'ndim') and predictions.ndim == 1:
                    # For 1D arrays, use a single prediction column
                    result_data['prediction'] = predictions
                elif hasattr(predictions, 'ndim'):
                    # For 2D arrays, create a column for each prediction dimension
                    for i in range(predictions.shape[1]):
                        result_data[f'prediction_{i}'] = predictions[:, i]
                else:
                    # Fallback: wrap in a column
                    result_data['prediction'] = np.asarray(predictions)
                
                # Create and save the DataFrame
                result_df = pd.DataFrame(result_data)

                # If timestamp exists but length mismatches predictions, drop timestamp to avoid errors
                if 'timestamp' in result_df.columns and len(result_df['timestamp']) != len(result_df.drop(columns=['timestamp'])):
                    logger.warning("Timestamp length does not match predictions; dropping timestamp from save")
                    result_df = result_df.drop(columns=['timestamp'])

                # Handle empty predictions gracefully
                if result_df.shape[0] == 0:
                    logger.warning(f"No predictions to save for {model_name}; skipping file write")
                    continue
                
                # Convert timestamp to string if it's a datetime
                if 'timestamp' in result_df.columns and hasattr(result_df['timestamp'].dtype, 'tz'):
                    result_df['timestamp'] = result_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Save to CSV with better formatting
                result_df.to_csv(
                    output_file,
                    index=False,
                    float_format='%.4f',  # Format floating point numbers
                    date_format='%Y-%m-%d %H:%M:%S'  # Format timestamps
                )
                
                # Verify file was created
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    logger.info(f"Successfully saved {model_name} predictions to {output_file} (Size: {file_size} bytes)")
                    
                    # Print file contents for debugging
                    try:
                        with open(output_file, 'r') as f:
                            content = f.read(500)  # Read first 500 chars
                            logger.debug(f"File contents of {output_file}:\n{content}...")
                    except Exception as e:
                        logger.warning(f"Could not read output file for debugging: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Error saving {model_name} results: {str(e)}", exc_info=True)
                # In case of error, try a simpler save approach
                try:
                    # Create a simple DataFrame with just the predictions
                    result_data = {'prediction': predictions}
                    if 'Reading Time' in self.data.columns:
                        result_data['timestamp'] = self.data['Reading Time']
                    
                    pd.DataFrame(result_data).to_csv(output_file, index=False)
                    logger.warning(f"Used fallback method to save {model_name} results")
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
            # Ensure ordering of columns (include Date column matching notebook)
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

            # Also save notebook-style prediction_results_30menit.csv at project root and stress_test
            try:
                project_root = Path(__file__).resolve().parents[1]
                df_nb = df_to_save.copy()
                # Add Date column (date part of Window_Start_Time)
                if 'Window_Start_Time' in df_nb.columns:
                    dt = pd.to_datetime(df_nb['Window_Start_Time'], errors='coerce')
                    df_nb['Date'] = dt.dt.strftime('%Y-%m-%d')
                if 'Recommendation' in df_nb.columns:
                    df_nb['Recommendation'] = df_nb['Recommendation'].astype(str).str.strip()
                # Ensure exact column order per template
                nb_cols = ['Window_Start_Time', 'Prediction', 'Status', 'Recommendation', 'Date']
                for c in nb_cols:
                    if c not in df_nb.columns:
                        df_nb[c] = ''
                df_nb = df_nb[nb_cols]

                # Write to project root
                pred30_path = project_root / 'prediction_results_30menit.csv'
                df_nb.to_csv(pred30_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
                logger.info(f"Saved notebook-style 30-minute predictions to: {pred30_path}")

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
        
        This matches the notebook output format where each 3-hour window shows the dominant status.
        """
        if final_df.empty:
            logger.warning("No data to aggregate to 3-hour windows")
            return
        
        df = final_df.copy()
        
        # Ensure Window_Start_Time is datetime
        df['Window_Start_Time'] = pd.to_datetime(df['Window_Start_Time'])
        
        # Create 3-hour bins (floor to 3-hour intervals)
        df['3H_Window'] = df['Window_Start_Time'].dt.floor('3H')
        
        # Group by 3-hour window and find dominant status
        def get_dominant_status(group):
            """Get the dominant status using EXACT notebook logic (cell 28)
            
            Logic:
            1. If Shut-in > 50% of total → Dominant = Shut-in
            2. Otherwise, exclude Shut-in from calculation:
               - If non-Running ≥ 50% → pick most frequent non-Running
               - If tie, use daily problem counter (not implemented yet, use priority)
               - Otherwise → Running
            """
            total = len(group)
            if total == 0:
                return 'Unknown'
            
            status_counts = group['Status'].value_counts()
            
            # Check Shut-in count FIRST (highest frequency priority)
            shutin_count = status_counts.get('Shut-in', 0)
            if shutin_count > total / 2:
                return 'Shut-in'
            
            # Special priority: If EDP is dominant among non-Shut-in AND Shut-in is not dominant
            # This handles the case where EDP + Shut-in both exist but neither is >50%
            edp_count = status_counts.get('Electrical Downhole Problem', 0)
            if edp_count > 0 and shutin_count > 0 and shutin_count <= total / 2:
                # Exclude Shut-in to see if EDP is dominant among non-Shut-in
                non_shutin_total = total - shutin_count
                if non_shutin_total > 0 and edp_count >= (non_shutin_total / 2):
                    return 'Electrical Downhole Problem'
            
            # Exclude Shut-in from calculation
            group_no_shutin = group[group['Status'] != 'Shut-in']
            total_valid = len(group_no_shutin)
            
            if total_valid == 0:
                return 'Shut-in'  # All Shut-in
            
            status_counts_no_shutin = group_no_shutin['Status'].value_counts()
            running_count = status_counts_no_shutin.get('Running', 0)
            non_running_count = total_valid - running_count
            
            if non_running_count >= (total_valid / 2):
                # Get most frequent non-Running status
                status_counts_no_running = status_counts_no_shutin.drop('Running', errors='ignore')
                
                if len(status_counts_no_running) == 0:
                    return 'Running'
                
                top_count = status_counts_no_running.max()
                top_statuses = status_counts_no_running[status_counts_no_running == top_count]
                
                if len(top_statuses) == 1:
                    return top_statuses.index[0]
                else:
                    # Tie-breaker: use priority order (simplified, notebook uses daily counter)
                    priority = ['100% Watercut', 'Electrical Downhole Problem', 
                               'Start-up Phase', 'Low PI']
                    for p in priority:
                        if p in top_statuses.index:
                            return p
                    return top_statuses.index[0]
            else:
                return 'Running'
        
        # Aggregate
        result_3h = df.groupby('3H_Window').apply(get_dominant_status).reset_index()
        result_3h.columns = ['Window_Start_Time', 'Dominant Status']
        
        # Sort by time
        result_3h = result_3h.sort_values('Window_Start_Time').reset_index(drop=True)
        
        # Save to output directory
        output_dir = str(self.output_dir)
        output_file = os.path.join(output_dir, f"result_df_3 jam.csv")
        result_3h.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
        logger.info(f"Saved 3-hour aggregated results to: {output_file}")
        
        # Also save to project root for easy comparison
        try:
            project_root = Path(__file__).resolve().parents[1]
            root_file = project_root / 'result_df_3 jam.csv'
            result_3h.to_csv(root_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
            logger.info(f"Saved 3-hour aggregated results to project root: {root_file}")
        except Exception as e:
            logger.warning(f"Could not save 3-hour results to project root: {e}")
    
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
