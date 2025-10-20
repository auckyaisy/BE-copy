from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
SRC_DIR = BASE_DIR / 'src'

# Input/Output paths
INPUT_DIR = DATA_DIR / 'input'
OUTPUT_DIR = DATA_DIR / 'output'

# Model paths (update these paths after placing your models)
MODEL_PATHS = {
    'discharge_pressure': MODEL_DIR / 'knn_dp_SKW.joblib',  # Using the provided model
    'virtual_rate': MODEL_DIR / 'knn_model_r.joblib',       # Using the provided model
    'failure_prediction': MODEL_DIR / 'knn_tanpa SMOTE_model.pkl'  # Using the provided model
}

# Default parameters
DEFAULT_PARAMS = {
    'knn_n_neighbors': 5,
    'test_size': 0.2,
    'random_state': 42
}

# Feature columns for each model
FEATURE_COLUMNS = {
    'discharge_pressure': [
        'Average Amps (A) (Raw)',
        'Intake Pressure (psi) (Raw)',
        'Intake Temperature (F) (Raw)',
        'Motor Temperature (F) (Raw)',
        'Vibration (gravit) (Raw)'
    ],
    'virtual_rate': [
        # Model VR expect 4 features: A, IP, IT, MT
        # Notebook line 180: df.drop(df.columns[[0,1,2,6]], axis=1) gives 6 cols
        # But model trained with 4 cols (lines 185-188 commented out show rename to A, IP, IT, MT)
        # So model uses: Amps, Intake Pressure, Intake Temp, Motor Temp
        'Average Amps (A) (Raw)',
        'Intake Pressure (psi) (Raw)',
        'Intake Temperature (F) (Raw)',
        'Motor Temperature (F) (Raw)'
    ],
    'failure_prediction': [
        'Average Amps (A) (Raw)',
        'Drive Frequency (Hz) (Raw)',
        'Intake Pressure (psi) (Raw)',
        'Discharge Pressure (psi) (Raw)',
        'Virtual Rate (BFPD) (Raw)',
        'Slope'
    ]
}

# Target columns for each model
TARGET_COLUMNS = {
    'discharge_pressure': 'Discharge Pressure (psi) (Raw)',
    'virtual_rate': 'Virtual Rate (BFPD) (Raw)',
    'failure_prediction': 'Failure'
}
