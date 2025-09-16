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
    
    def __init__(self, well_name: str):
        """
        Initialize the pipeline with a specific well name.
        
        Args:
            well_name: Name of the well (used for input/output file naming)
        """
        self.well_name = well_name
        self.models = {}
        self.data = None
        self.predictions = {}
        self.df_wc: Optional[pd.DataFrame] = None
        
        # Create output directory if it doesn't exist
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
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
            
            # Handle missing values
            if isinstance(X, pd.DataFrame) and X.isna().any().any():
                logger.warning("Input contains NaN values. Imputing with column means.")
                
                # Create a mask to remember which columns had NaNs
                cols_with_nan = X.columns[X.isna().any()].tolist()
                
                if cols_with_nan:
                    # Initialize imputer with mean strategy
                    imputer = SimpleImputer(strategy='mean')
                    
                    # Only transform columns with NaNs
                    X_imputed = imputer.fit_transform(X[cols_with_nan])
                    
                    # Update the original DataFrame with imputed values
                    if isinstance(X, pd.DataFrame):
                        X[cols_with_nan] = X_imputed
                    else:
                        X = np.hstack([X, X_imputed]) if X.ndim > 1 else X_imputed.reshape(-1)
            
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
            # Define the mapping between our column names and the model's expected feature names
            feature_mapping = {
                'A': 'Average Amps (A) (Raw)',
                'IP': 'Intake Pressure (psi) (Raw)',
                'DP': 'Discharge Pressure (psi) (Raw)',
                'IT': 'Intake Temperature (F) (Raw)',
                'MT': 'Motor Temperature (F) (Raw)',
                'V': 'Vibration (gravit) (Raw)',
                'R': 'Virtual Rate (BFPD) (Raw)'  # R stands for Rate
            }
            
            # Create a new DataFrame with the expected column names
            features = pd.DataFrame()
            
            # Map the columns from the input data to the expected feature names
            for model_col, data_col in feature_mapping.items():
                if data_col in data.columns:
                    features[model_col] = data[data_col]
                else:
                    # If the column is missing, fill with zeros and log a warning
                    features[model_col] = 0.0
                    logger.warning(f"Filled missing column '{data_col}' with zeros")
            
            # Ensure we have all expected columns in the correct order
            expected_cols = ['A', 'IP', 'DP', 'IT', 'MT', 'V', 'R']
            for col in expected_cols:
                if col not in features.columns:
                    features[col] = 0.0
                    logger.warning(f"Added missing expected column '{col}' with zeros")
            
            # Reorder columns to match the expected order
            features = features[expected_cols]
            
        # For other models, use the standard feature extraction
        else:
            # Get the required feature columns for this model
            required_cols = FEATURE_COLUMNS[model_name]
            
            # Log available and required columns for debugging
            logger.debug(f"Available columns in data: {data.columns.tolist()}")
            logger.debug(f"Required columns for {model_name}: {required_cols}")
            
            # Check which required columns are missing
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                # Log a warning about missing columns but continue with available ones
                logger.warning(f"Missing columns for {model_name}: {missing_cols}. Using available columns.")
                
                # Only keep columns that exist in the data
                available_cols = [col for col in required_cols if col in data.columns]
                
                if not available_cols:
                    raise ValueError(f"No required columns found in data for {model_name}")
                    
                logger.info(f"Using available columns for {model_name}: {available_cols}")
                features = data[available_cols].copy()
                
                # Fill missing columns with zeros (you might want to use a different strategy)
                for col in missing_cols:
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
            # Load the data
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
            discharge_pressure = self.predict('discharge_pressure')
            results['discharge_pressure'] = discharge_pressure
            self.data['Discharge Pressure (psi) (Raw)'] = (
                self.data.get('Discharge Pressure (psi) (Raw)', pd.Series(index=self.data.index))
            )
            # store predicted to a separate column for downstream visibility
            self.data['predicted_discharge_pressure'] = discharge_pressure

            # 2. Virtual Rate Prediction (Virtual Rate (BFPD) (Raw))
            logger.info("2/4 - Running virtual rate prediction...")
            virtual_rate = self.predict('virtual_rate')
            results['virtual_rate'] = virtual_rate
            self.data['Virtual Rate (BFPD) (Raw)'] = virtual_rate

            # Apply template rule: if Amps==0 and Freq==0 then Virtual Rate = 0
            if {'Average Amps (A) (Raw)', 'Drive Frequency (Hz) (Raw)'}.issubset(self.data.columns):
                zero_mask = (
                    (self.data['Average Amps (A) (Raw)'].fillna(0) == 0)
                    & (self.data['Drive Frequency (Hz) (Raw)'].fillna(0) == 0)
                )
                self.data.loc[zero_mask, 'Virtual Rate (BFPD) (Raw)'] = 0.0

            # 3. Resample to 30-minute grid (df_all equivalent)
            logger.info("3/4 - Building 30-minute resampled dataset (df_all)...")
            df_all = self._build_df_all_30min(self.data)

            # 4. Compute per-window slopes over 30-minute windows (slopes_df equivalent)
            logger.info("4/4 - Computing 30-minute window slopes...")
            slopes_df = self._compute_window_slopes_30min(self.data)

            # Prepare slope features (df11 equivalent): A, IP, DP, IT, MT, V, R
            expected_cols = ['A', 'IP', 'DP', 'IT', 'MT', 'V', 'R']
            # Some windows may be missing columns; create safely
            if slopes_df.empty:
                df11 = pd.DataFrame(columns=expected_cols)
            else:
                missing = [c for c in expected_cols if c not in slopes_df.columns]
                tmp = slopes_df.copy()
                for c in missing:
                    tmp[c] = 0.0
                df11 = tmp[expected_cols].copy()

            # 5. Failure Prediction on windowed slope features
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
            df_idx.resample('30T', origin='epoch')
                 .mean(numeric_only=True)
                 .reset_index()
        )
        # Ensure column name matches template
        if 'Reading Time' not in df_resampled.columns:
            df_resampled = df_resampled.rename(columns={'index': 'Reading Time'})
        return df_resampled

    def _compute_window_slopes_30min(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute linear slopes per 30-minute window for specific numeric columns
        and return a DataFrame with columns: Window_Start_Time, A, IP, DP, IT, MT, V, R.
        Slopes are computed via numpy.polyfit against seconds from window start.
        """
        if 'Reading Time' not in df.columns:
            raise ValueError("'Reading Time' column is required for slope computation")

        use_cols_map = {
            'A': 'Average Amps (A) (Raw)',
            'IP': 'Intake Pressure (psi) (Raw)',
            'DP': 'Discharge Pressure (psi) (Raw)',
            'IT': 'Intake Temperature (F) (Raw)',
            'MT': 'Motor Temperature (F) (Raw)',
            'V': 'Vibration (gravit) (Raw)',
            'R': 'Virtual Rate (BFPD) (Raw)',
        }

        dfw = df.copy()
        dfw['Reading Time'] = pd.to_datetime(dfw['Reading Time'], errors='coerce')
        dfw = dfw.dropna(subset=['Reading Time']).sort_values('Reading Time').reset_index(drop=True)

        # Windows aligned to :00 and :30
        start_time = dfw['Reading Time'].iloc[0].floor('30min')
        end_time = dfw['Reading Time'].iloc[-1].ceil('30min')
        window_starts = pd.date_range(start=start_time, end=end_time, freq='30min')

        rows: List[Dict[str, Union[float, pd.Timestamp]]] = []
        for ws in window_starts:
            we = ws + pd.Timedelta(minutes=30)
            wd = dfw[(dfw['Reading Time'] >= ws) & (dfw['Reading Time'] < we)]
            if len(wd) < 2:
                # need at least 2 points for slope
                continue

            slopes_row: Dict[str, Union[float, pd.Timestamp]] = {'Window_Start_Time': ws}
            # seconds from window start
            tsec = (wd['Reading Time'] - ws).dt.total_seconds().values
            for short, col in use_cols_map.items():
                if col not in wd.columns:
                    slopes_row[short] = np.nan
                    continue
                series = wd[[col]].dropna().copy()
                if series.empty:
                    slopes_row[short] = np.nan
                    continue
                # align time with non-null rows
                mask_nonnull = wd[col].notna().values
                x = tsec[mask_nonnull]
                y = wd[col].dropna().values
                if y.size < 2 or x.size < 2:
                    slopes_row[short] = np.nan
                    continue
                # slope via polyfit (slope per second). Convert to per-minute to mirror templates where needed
                try:
                    m, _ = np.polyfit(x, y, 1)
                    slopes_row[short] = float(m)  # keep as per-second slope; rules use ~0 tolerance
                except Exception:
                    slopes_row[short] = np.nan

            rows.append(slopes_row)

        slopes_df = pd.DataFrame(rows)
        # Ensure types
        if not slopes_df.empty:
            slopes_df['Window_Start_Time'] = pd.to_datetime(slopes_df['Window_Start_Time'])
        return slopes_df

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
            }.get(x, 'Unidentified')

        def recommendation_map(x: int) -> str:
            x = int(x)
            if x == 0:
                return ""
            # keep others as empty strings to mirror template table output
            return ""

        out['Status'] = out['Prediction'].apply(status_map)
        out['Recommendation'] = out['Prediction'].apply(recommendation_map)

        # Apply additional rules using df_all and slopes_df akin to the template
        try:
            TOL = 1e-8
            df_all2 = df_all.copy()
            df_all2['Reading Time'] = pd.to_datetime(df_all2['Reading Time'], errors='coerce')
            slopes_df2 = slopes_df.copy()
            slopes_df2['Window_Start_Time'] = pd.to_datetime(slopes_df2['Window_Start_Time'], errors='coerce')

            for idx, row in out.iterrows():
                ws = row['Window_Start_Time']

                # 100% Watercut rule from daily production file, if present
                try:
                    if self.df_wc is not None and not self.df_wc.empty:
                        wc_row = self.df_wc[self.df_wc['Date'] == ws.normalize()]
                        if not wc_row.empty:
                            wc_val = wc_row['WC'].iloc[0]
                            if pd.notna(wc_val) and np.isclose(float(wc_val), 100.0, atol=1e-6):
                                out.at[idx, 'Prediction'] = 12
                                out.at[idx, 'Status'] = '100% Watercut'
                                out.at[idx, 'Recommendation'] = ''
                                continue
                except Exception as _:
                    pass
                match_row = df_all2[df_all2['Reading Time'] == ws]
                slope_row = slopes_df2[slopes_df2['Window_Start_Time'] == ws]
                if match_row.empty or slope_row.empty:
                    continue

                amps = match_row.get('Average Amps (A) (Raw)', pd.Series([np.nan])).iloc[0]
                freq = match_row.get('Drive Frequency (Hz) (Raw)', pd.Series([np.nan])).iloc[0]
                rate = match_row.get('Virtual Rate (BFPD) (Raw)', pd.Series([np.nan])).iloc[0]

                dp = slope_row.get('DP', pd.Series([np.nan])).iloc[0]
                it = slope_row.get('IT', pd.Series([np.nan])).iloc[0]
                mt = slope_row.get('MT', pd.Series([np.nan])).iloc[0]
                v_ = slope_row.get('V', pd.Series([np.nan])).iloc[0]
                r_ = slope_row.get('R', pd.Series([np.nan])).iloc[0]

                # Check 30-minute window variation in original data
                end_time = ws + pd.Timedelta(minutes=30)
                subset_ori = self.data[(pd.to_datetime(self.data['Reading Time'], errors='coerce') >= ws) &
                                       (pd.to_datetime(self.data['Reading Time'], errors='coerce') < end_time)]
                cols_check = [
                    'Intake Pressure (psi) (Raw)',
                    'Discharge Pressure (psi) (Raw)',
                    'Intake Temperature (F) (Raw)',
                    'Motor Temperature (F) (Raw)',
                    'Vibration (gravit) (Raw)',
                    'Virtual Rate (BFPD) (Raw)'
                ]
                has_variation = False
                if not subset_ori.empty:
                    for c in cols_check:
                        if c in subset_ori.columns and subset_ori[c].nunique(dropna=True) > 1:
                            has_variation = True
                            break

                # Shut-in rule
                if np.isclose(amps if pd.notna(amps) else 0, 0, atol=TOL) and np.isclose(freq if pd.notna(freq) else 0, 0, atol=TOL):
                    other_cols = [
                        'Virtual Rate (BFPD) (Raw)',
                        'Discharge Pressure (psi) (Raw)',
                        'Intake Temperature (F) (Raw)',
                        'Motor Temperature (F) (Raw)',
                        'Vibration (gravit) (Raw)'
                    ]
                    all_zero = True
                    for c in other_cols:
                        val = match_row.get(c, pd.Series([np.nan])).iloc[0]
                        if not np.isclose(val if pd.notna(val) else 0, 0, atol=TOL):
                            all_zero = False
                            break

                    if all_zero or has_variation:
                        out.at[idx, 'Prediction'] = 11
                        out.at[idx, 'Status'] = 'Shut-in'
                        out.at[idx, 'Recommendation'] = ''
                        continue

                # EDP rule (all near-zero and no variation)
                conditions = [
                    np.isclose(amps if pd.notna(amps) else 0, 0, atol=TOL),
                    np.isclose(freq if pd.notna(freq) else 0, 0, atol=TOL),
                    np.isclose(rate if pd.notna(rate) else 0, 0, atol=TOL),
                    np.isclose(dp if pd.notna(dp) else 0, 0, atol=TOL),
                    np.isclose(it if pd.notna(it) else 0, 0, atol=TOL),
                    np.isclose(mt if pd.notna(mt) else 0, 0, atol=TOL),
                    np.isclose(v_ if pd.notna(v_) else 0, 0, atol=TOL),
                    np.isclose(r_ if pd.notna(r_) else 0, 0, atol=TOL),
                ]
                if all(conditions) and not has_variation:
                    out.at[idx, 'Prediction'] = 10
                    out.at[idx, 'Status'] = 'Electrical Downhole Problem'
                    out.at[idx, 'Recommendation'] = ''

        except Exception as e:
            logger.warning(f"Failed applying additional status rules: {e}")

        return out

    def _load_wc_data(self) -> None:
        """Load daily Watercut data from prod_data.csv located at project root.
        Parses 'Date' to datetime (normalized to date) and 'WC' to numeric percent.
        """
        project_root = Path(__file__).resolve().parents[1]
        csv_path = project_root / 'prod_data.csv'
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
    
    def _save_results(self, results: Dict[str, np.ndarray]) -> None:
        """Save prediction results to CSV files.
        
        Args:
            results: Dictionary containing the prediction results
        """
        # Use configured OUTPUT_DIR (data/output)
        output_dir = str(OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving results to directory: {output_dir}")
        
        for model_name, predictions in results.items():
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
                if predictions.ndim == 1:
                    # For 1D arrays, use a single prediction column
                    result_data['prediction'] = predictions
                else:
                    # For 2D arrays, create a column for each prediction dimension
                    for i in range(predictions.shape[1]):
                        result_data[f'prediction_{i}'] = predictions[:, i]
                
                # Create and save the DataFrame
                result_df = pd.DataFrame(result_data)
                
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
        output_dir = str(OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{self.well_name}_failure_prediction_30min.csv")
        try:
            # Ensure ordering of columns
            cols = ['Window_Start_Time', 'Prediction', 'Status', 'Recommendation']
            df_to_save = final_df[cols].copy() if all(c in final_df.columns for c in cols) else final_df.copy()
            df_to_save.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
            logger.info(f"Saved final failure results to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving final failure results: {e}")

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

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
            dp_plot = OUTPUT_DIR / f"{self.well_name}_discharge_pressure_plot.png"
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
            vr_plot = OUTPUT_DIR / f"{self.well_name}_virtual_rate_plot.png"
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
            overview_plot = OUTPUT_DIR / f"{self.well_name}_overview_plot.png"
            fig3.savefig(overview_plot, dpi=150)

            # Try to open saved plots
            for p in [dp_plot, vr_plot, overview_plot]:
                self._open_in_os(p)

            # Also show interactively if environment allows
            try:
                plt.show()
            except Exception:
                pass

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
