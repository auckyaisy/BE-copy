
### Basic Usage

```bash
python3 main.py --well-name SKW-07
```

### Full Options

```bash
python3 main.py \
  --well-name SKW-33 \
  --prod-data "/Users/m/Downloads/mba_sivly/BE-copy/data/input/Data Produksi/SKW-33.csv" \
  --input-file "/Users/m/Downloads/mba_sivly/BE-copy/data/input/Data Sensor/SKW-33.csv" \
  --output-dir "data/output/SKW-33" \
  --log-level INFO
```

## Arguments

### Required
- `--well-name`: Name of the well (e.g., SKW-07, SKW-18)

### Optional
- `--input-file`: Path to sensor data CSV
  - Default: `data/input/{well_name}.csv`
  - Example: `data/input/Data Sensor/SKW-07.csv`

- `--prod-data`: Path to production/watercut data CSV
  - Default: `prod_data.csv` in project root
  - Example: `data/input/Data Produksi/SKW-07.csv`
  - Format:
    ```csv
    Date,WC
    2025-01-07,100.00
    2025-01-08,85.50
    ```

- `--output-dir`: Output directory for results
  - Default: `data/output`
  - Example: `data/output/SKW-07`

- `--model`: Which model to run
  - Choices: `all`, `discharge_pressure`, `virtual_rate`, `slope`, `failure_prediction`
  - Default: `all`

- `--log-level`: Logging verbosity
  - Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
  - Default: `INFO`

## Examples

### Example 1: Basic run with default paths
```bash
# Assumes data/input/SKW-07.csv exists
python3 main.py --well-name SKW-07
```

### Example 2: With custom sensor data path
```bash
python3 main.py \
  --well-name SKW-07 \
  --input-file "data/input/Data Sensor/SKW-07.csv"
```

### Example 3: With both sensor and production data
```bash
python3 main.py \
  --well-name SKW-07 \
  --input-file "data/input/Data Sensor/SKW-07.csv" \
  --prod-data "data/input/Data Produksi/SKW-07.csv" \
  --output-dir "data/output/SKW-07"
```

### Example 4: Debug mode
```bash
python3 main.py \
  --well-name SKW-07 \
  --input-file "data/input/Data Sensor/SKW-07.csv" \
  --log-level DEBUG
```

### Example 5: Run specific model only
```bash
# Only run failure prediction
python3 main.py \
  --well-name SKW-07 \
  --model failure_prediction
```

## Batch Processing (run_all_wells.py)

Process all wells in `data/input/Data Sensor/`:

```bash
python3 run_all_wells.py
```

This will:
1. Find all CSV files in `data/input/Data Sensor/`
2. For each well, check if corresponding prod_data exists in `data/input/Data Produksi/`
3. Run full analysis for each well
4. Save outputs to `data/output/{well_name}/`

## Output Files

For each well, the following files are generated:

### CSV Files
- `{well_name}_discharge_pressure_predictions.csv` - Discharge pressure predictions
- `{well_name}_virtual_rate_predictions.csv` - Virtual rate predictions
- `{well_name}_failure_prediction_30min.csv` - **Main output** with failure predictions and status

### Plot Files
- `{well_name}_discharge_pressure_plot.png` - Discharge pressure visualization
- `{well_name}_virtual_rate_plot.png` - Virtual rate visualization
- `{well_name}_overview_plot.png` - Combined overview plot

### Additional Files (in project root)
- `df_all.csv` - Resampled 30-minute data
- `slopes_df_30menit.csv` - Calculated slopes
- `prediction_results_30menit.csv` - Prediction results (notebook format)
- `X_predict_30menit.csv` - Feature matrix
- `failure_features_used_30menit.csv` - Features used for prediction

## Status Classes

The pipeline predicts 14 different status classes:

| Class | Status | Description |
|-------|--------|-------------|
| 0 | Running | Normal operation |
| 1 | Low PI | Low productivity index |
| 2 | Pump Wear | Pump degradation |
| 3 | Tubing Leak | Tubing integrity issue |
| 4 | Higher PI | Increased productivity |
| 5 | Increase in Frequency | Frequency adjustment |
| 6 | Open Choke | Choke position change |
| 7 | Increase in Watercut | Rising water content |
| 8 | Sand Ingestion | Sand production |
| 9 | Closed Valve | Valve restriction |
| 10 | Electrical Downhole Problem | EDP detected |
| 11 | Shut-in | Well shut-in |
| 12 | 100% Watercut | Complete water production |
| 13 | Start-up Phase | Post-gap startup period |

## Override Rules

The pipeline applies these rules in order:

1. **Watercut Override**: If WC=100% in prod_data → Class 12
2. **EDP Override**: If Amps=0, Freq=0, no variation → Class 10
3. **Shut-in Override**: If Amps≈0, Freq≈0, with variation → Class 11
4. **Start-up Phase**: If gap >3h in resampled data → Class 13

## Troubleshooting

### File not found error
```bash
# Make sure file exists
ls -la "data/input/Data Sensor/SKW-07.csv"

# Use absolute path if needed
python3 main.py \
  --well-name SKW-07 \
  --input-file "/full/path/to/SKW-07.csv"
```

### Watercut override not working
```bash
# Specify prod_data explicitly
python3 main.py \
  --well-name SKW-07 \
  --prod-data "data/input/Data Produksi/SKW-07.csv"

# Check prod_data format
head data/input/Data Produksi/SKW-07.csv
```

### View detailed logs
```bash
# Use DEBUG level
python3 main.py --well-name SKW-07 --log-level DEBUG

# Check log file
cat logs/SKW-07_analysis.log
```
