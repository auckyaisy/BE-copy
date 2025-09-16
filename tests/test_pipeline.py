"""
Test script for the Well Analysis Pipeline.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import WellAnalysisPipeline
from config.config import INPUT_DIR, OUTPUT_DIR

def create_test_data(num_samples=100):
    """Create test data for the pipeline with exact column names as expected by the models.
    
    The data includes all required columns for the complete workflow:
    1. Discharge Pressure Prediction
    2. Virtual Rate Prediction
    3. Slope Calculation
    4. Failure Prediction
    """
    # Generate timestamps
    timestamps = pd.date_range(start='2023-01-01', periods=num_samples, freq='h')
    
    # Create base features with exact column names
    data = {
        # Metadata
        'Location': ['SKW-33'] * num_samples,
        'UWI': ['SKW-33'] * num_samples,
        'Reading Time': timestamps,
        
        # For Discharge Pressure Prediction
        'Average Amps (A) (Raw)': np.random.uniform(5, 15, num_samples),
        # Removed 'Drive Frequency (Hz) (Raw)' as it's causing issues
        'Intake Pressure (psi) (Raw)': np.random.uniform(100, 500, num_samples),
        'Intake Temperature (F) (Raw)': np.random.uniform(80, 200, num_samples),
        'Motor Temperature (F) (Raw)': np.random.uniform(100, 250, num_samples),
        'Vibration (gravit) (Raw)': np.random.uniform(0, 1, num_samples),
        
        # For Virtual Rate Prediction
        'Discharge Pressure (psi) (Raw)': np.random.uniform(500, 2000, num_samples),
        
        # For Failure Prediction
        'Virtual Rate (BFPD) (Raw)': np.random.uniform(0, 5000, num_samples),
        'Slope': np.random.normal(0, 0.1, num_samples)
    }
    
    # Add some trend to make the data more realistic
    trend = np.linspace(0, 10, num_samples)
    data['Drive Frequency (Hz) (Raw)'] = 50 + 10 * np.sin(trend) + np.random.normal(0, 2, num_samples)
    data['Intake Pressure (psi) (Raw)'] = 300 + 100 * np.sin(trend) + np.random.normal(0, 20, num_samples)
    data['Discharge Pressure (psi) (Raw)'] = 1000 + 500 * np.sin(trend) + np.random.normal(0, 50, num_samples)
    
    # Ensure all values are positive
    for col in data:
        if '(Raw)' in col and col != 'Reading Time':
            data[col] = np.abs(data[col])
    
    df = pd.DataFrame(data)
    
    # Ensure the 'Reading Time' column is in datetime format
    df['Reading Time'] = pd.to_datetime(df['Reading Time'])
    
    return df

def test_pipeline():
    """Test the WellAnalysisPipeline with sample data."""
    # Create test data
    test_data = create_test_data(100)
    
    # Ensure input directory exists
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save test data
    test_file = INPUT_DIR / 'test_well.csv'
    test_data.to_csv(test_file, index=False)
    print(f"Saved test data to {test_file}")
    
    try:
        # Initialize and run the pipeline
        print("Initializing pipeline...")
        pipeline = WellAnalysisPipeline('test_well')
        
        print("Running full analysis...")
        results = pipeline.run_full_analysis(test_file)
        
        print("\nAnalysis completed successfully!")
        print("\nResults summary:")
        for model_name, predictions in results.items():
            print(f"- {model_name}: {len(predictions)} predictions")
        
        # Check output files
        output_files = list(OUTPUT_DIR.glob('test_well_*'))
        print(f"\nGenerated {len(output_files)} output files in {OUTPUT_DIR}:")
        for f in output_files:
            print(f"- {f.name}")
            
        return True
        
    except Exception as e:
        print(f"Error during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up test files
        if test_file.exists():
            os.remove(test_file)
        
        # Clean up output files
        for f in OUTPUT_DIR.glob('test_well_*'):
            try:
                os.remove(f)
            except:
                pass

if __name__ == "__main__":
    test_pipeline()
