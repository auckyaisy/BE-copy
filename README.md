# Well Analysis Pipeline

A Python-based pipeline for analyzing well data and making predictions using machine learning models. This pipeline integrates multiple models for discharge pressure prediction, virtual rate calculation, slope analysis, and failure prediction.

## Features

- **Modular Design**: Easily extendable architecture for adding new models and analysis steps
- **Data Preprocessing**: Handles missing data, feature engineering, and time series resampling
- **Model Integration**: Supports loading and using pre-trained models
- **Visualization**: Generates plots for model predictions and analysis results
- **Command-line Interface**: Simple CLI for running analyses on different wells

## Project Structure

```
well_analysis_pipeline/
├── config/                 # Configuration files
│   └── config.py           # Main configuration
├── data/                   # Data directories
│   ├── input/              # Input CSV files (one per well)
│   └── output/             # Output files (predictions, plots)
├── models/                 # Pre-trained model files
├── src/                    # Source code
│   ├── pipeline.py         # Main pipeline implementation
│   └── utils.py            # Utility functions
├── main.py                 # Command-line interface
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd well_analysis_pipeline
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Preparing Input Data

1. Place your well data CSV files in the `data/input/` directory with the naming convention `{well_name}.csv`.
2. Ensure your CSV files contain the required columns (see Configuration section).

### Running the Pipeline

To run the full analysis pipeline for a specific well:

```bash
python main.py --well-name SKW-02 --model all
```

### Command-line Arguments

- `--well-name`: Name of the well (required)
- `--input-file`: Path to the input CSV file (optional, defaults to `data/input/{well_name}.csv`)
- `--output-dir`: Custom output directory (optional). If not provided, outputs are saved under `data/output/`.
- `--model`: Which model(s) to run (default: 'all')
  - Options: 'all', 'discharge_pressure', 'virtual_rate', 'slope', 'failure_prediction'
- `--log-level`: Logging level (default: 'INFO')
  - Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

### Example

```bash
# Run full analysis with debug logging
python main.py --well-name SKW-02 --model all --log-level DEBUG

# Run only discharge pressure prediction
python main.py --well-name SKW-02 --model discharge_pressure

# Run full analysis and write outputs to a custom folder
python main.py --well-name SKW-02 --model all --output-dir ./results/SKW-02
```

The pipeline expects CSV files with the following columns (adjust in `config/config.py` if needed):

- `Reading Time`: Timestamp of the measurement
- `Average Amps (A) (Raw)`: Average current
- `Drive Frequency (Hz) (Raw)`: Drive frequency
- `Intake Pressure (psi) (Raw)`: Intake pressure
- `Discharge Pressure (psi) (Raw)`: Discharge pressure (target for prediction)
- `Intake Temperature (F) (Raw)`: Intake temperature
- `Motor Temperature (F) (Raw)`: Motor temperature
- `Vibration (gravit) (Raw)`: Vibration measurement

### Model Configuration

Model paths and parameters can be configured in `config/config.py`. Update the following sections as needed:

```python
# Model paths (update these paths after placing your models)
MODEL_PATHS = {
    'discharge_pressure': MODEL_DIR / 'discharge_pressure_model.joblib',
    'virtual_rate': MODEL_DIR / 'virtual_rate_model.joblib',
    'slope': MODEL_DIR / 'slope_model.joblib',
    'failure_prediction': MODEL_DIR / 'failure_prediction_model.joblib'
}

# Feature columns for each model
FEATURE_COLUMNS = {
    'discharge_pressure': [
        'Average Amps (A) (Raw)',
        'Drive Frequency (Hz) (Raw)',
        'Intake Pressure (psi) (Raw)',
        'Intake Temperature (F) (Raw)',
        'Motor Temperature (F) (Raw)',
        'Vibration (gravit) (Raw)'
    ],
    # Add feature columns for other models
}
```

## Model Training (Optional)

To train new models, you can use the provided Jupyter notebooks in the `notebooks/` directory:

1. `Template Discharge Pressure.ipynb`
2. `Template Virtual Rate.ipynb`
3. `Template Slope_30 Menit.ipynb`
4. `Template Failure Pred_30 Menit.ipynb`

After training, save the models to the `models/` directory using `joblib.dump()`.

## Output

By default, the pipeline writes outputs to `data/output/`. You can override this with `--output-dir <path>`.

The following output files are produced in the chosen output directory:

- `{well_name}_{model_name}_predictions.csv`: CSV file containing the model predictions
- `{well_name}_{model_name}_plot.png`: Plot of the predictions vs. actual values (if available)
- `logs/{well_name}_analysis.log`: Log file with detailed execution information

## Build Executables (macOS and Windows)

You can package this project into a single-file executable for macOS and Windows using PyInstaller. The executable will:

- Accept the same CLI arguments as `python main.py`.
- Run the pipeline and save CSVs into the specified output directory (default `data/output/`).
- Generate plots and attempt to auto-open them in your OS default viewer.

### 1) Install PyInstaller

It is recommended to install PyInstaller into your virtual environment.

```bash
pip install pyinstaller
```

### 2) Build on macOS

Run the following from the project root:

```bash
pyinstaller --onefile --name well-analysis \
  --add-data "config:config" \
  --add-data "data:data" \
  --add-data "models:models" \
  main.py
```

Notes:

- `--add-data` ensures your `config/`, `data/`, and `models/` folders are bundled or available to the executable.
- On macOS, the resulting binary is created at `dist/well-analysis`.

Run it:

```bash
./dist/well-analysis --well-name SKW-18 --model all
```

### 3) Build on Windows

In Windows PowerShell or Command Prompt, run:

```powershell
pyinstaller --onefile --name well-analysis ^
  --add-data "config;config" ^
  --add-data "data;data" ^
  --add-data "models;models" ^
  main.py
```

Run it:

```powershell
dist\well-analysis.exe --well-name SKW-18 --model all
```

### 4) Optional: Reduce console window on Windows

If you prefer a windowless executable on Windows (still writes logs and opens plots), add `--noconsole`:

```powershell
pyinstaller --onefile --noconsole --name well-analysis ^
  --add-data "config;config" ^
  --add-data "data;data" ^
  --add-data "models;models" ^
  main.py
```

### 5) Passing input file directly

If your input is not located at `data/input/{well_name}.csv`, provide `--input-file`:

```bash
./dist/well-analysis --well-name SKW-18 --input-file path/to/SKW-18.csv --model all
```

### 6) What you’ll see at the end

- CSVs created in your chosen output directory (default `data/output/`):
  - `{well}_discharge_pressure_predictions.csv`
  - `{well}_virtual_rate_predictions.csv`
  - `{well}_failure_prediction_30min.csv`
- Plots auto-opened by the OS and saved under the chosen output directory:
  - `{well}_discharge_pressure_plot.png`
  - `{well}_virtual_rate_plot.png`
  - `{well}_overview_plot.png` (sensor overview + predictions)

If plots do not auto-open (e.g. on headless servers), you can open the PNGs manually from the output directory you specified.

## Troubleshooting

- **Missing Dependencies**: Ensure all required packages are installed using `pip install -r requirements.txt`
- **File Not Found**: Check that the input file exists at the specified path
- **Model Loading Errors**: Verify that the model files exist in the `models/` directory
- **Memory Issues**: For large datasets, consider processing the data in smaller chunks

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
