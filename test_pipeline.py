#!/usr/bin/env python3
"""
Test script to verify pipeline.py outputs match notebook outputs exactly.
Compares all intermediate and final outputs for SKW-02.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.pipeline import WellAnalysisPipeline
from src.utils import setup_logging

# Setup logging
setup_logging(log_file=Path('logs') / 'test_pipeline.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_dataframes(df1, df2, name, tolerance=1e-6):
    """Compare two dataframes and report differences."""
    print(f"\n{'='*80}")
    print(f"COMPARING: {name}")
    print(f"{'='*80}")
    
    # Check shapes
    if df1.shape != df2.shape:
        print(f"❌ SHAPE MISMATCH!")
        print(f"   Pipeline: {df1.shape}")
        print(f"   Notebook: {df2.shape}")
        return False
    else:
        print(f"✅ Shape matches: {df1.shape}")
    
    # Check columns
    pipeline_cols = set(df1.columns)
    notebook_cols = set(df2.columns)
    
    if pipeline_cols != notebook_cols:
        print(f"❌ COLUMN MISMATCH!")
        missing_in_pipeline = notebook_cols - pipeline_cols
        extra_in_pipeline = pipeline_cols - notebook_cols
        if missing_in_pipeline:
            print(f"   Missing in pipeline: {missing_in_pipeline}")
        if extra_in_pipeline:
            print(f"   Extra in pipeline: {extra_in_pipeline}")
        return False
    else:
        print(f"✅ Columns match: {len(df1.columns)} columns")
    
    # Compare values for common columns
    all_match = True
    for col in df1.columns:
        if col in df2.columns:
            # Handle datetime columns
            if pd.api.types.is_datetime64_any_dtype(df1[col]) or pd.api.types.is_datetime64_any_dtype(df2[col]):
                df1_dt = pd.to_datetime(df1[col], errors='coerce')
                df2_dt = pd.to_datetime(df2[col], errors='coerce')
                if not df1_dt.equals(df2_dt):
                    mismatches = (df1_dt != df2_dt).sum()
                    print(f"   ⚠️  Column '{col}': {mismatches} datetime mismatches")
                    all_match = False
            # Handle numeric columns
            elif pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                # Use np.allclose for floating point comparison
                mask1 = df1[col].notna()
                mask2 = df2[col].notna()
                
                # Check NaN patterns match
                if not mask1.equals(mask2):
                    nan_diff = (mask1 != mask2).sum()
                    print(f"   ⚠️  Column '{col}': {nan_diff} NaN pattern mismatches")
                    all_match = False
                    continue
                
                # Compare non-NaN values
                if mask1.any():
                    vals1 = df1.loc[mask1, col].values
                    vals2 = df2.loc[mask2, col].values
                    
                    if not np.allclose(vals1, vals2, rtol=tolerance, atol=tolerance, equal_nan=True):
                        max_diff = np.abs(vals1 - vals2).max()
                        mean_diff = np.abs(vals1 - vals2).mean()
                        print(f"   ⚠️  Column '{col}': numeric mismatch (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e})")
                        all_match = False
            # Handle string/object columns
            else:
                if not df1[col].equals(df2[col]):
                    mismatches = (df1[col] != df2[col]).sum()
                    print(f"   ⚠️  Column '{col}': {mismatches} value mismatches")
                    # Show first few mismatches
                    mismatch_idx = df1[col] != df2[col]
                    if mismatch_idx.any():
                        print(f"      First mismatch at index {mismatch_idx.idxmax()}:")
                        print(f"         Pipeline: {df1.loc[mismatch_idx.idxmax(), col]}")
                        print(f"         Notebook: {df2.loc[mismatch_idx.idxmax(), col]}")
                    all_match = False
    
    if all_match:
        print(f"✅ ALL VALUES MATCH!")
        return True
    else:
        print(f"❌ SOME VALUES DIFFER")
        return False


def test_skw02():
    """Test pipeline with SKW-02 data and compare with notebook outputs."""
    
    print("\n" + "="*80)
    print("TESTING PIPELINE WITH SKW-02")
    print("="*80)
    
    # Paths
    test_dir = Path("Test Web")
    sensor_file = test_dir / "Data Sensor" / "SKW-02.csv"
    prod_file = test_dir / "Data Produksi" / "SKW-02.csv"
    notebook_dir = test_dir / "Hasil Bacaan Notebook" / "SKW-02"
    
    # Copy prod_data to project root (needed for watercut)
    import shutil
    shutil.copy(prod_file, Path("prod_data.csv"))
    
    # Initialize pipeline
    well_name = "SKW-02"
    output_dir = Path("test_output") / well_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = WellAnalysisPipeline(well_name, output_dir=output_dir)
    
    print(f"\n📂 Loading data from: {sensor_file}")
    
    # Run pipeline
    try:
        results = pipeline.run_full_analysis(input_file=sensor_file)
        print("\n✅ Pipeline completed successfully!")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare outputs
    print("\n" + "="*80)
    print("COMPARING OUTPUTS WITH NOTEBOOK RESULTS")
    print("="*80)
    
    all_tests_passed = True
    
    # 1. Compare SKW Final.csv (after DP prediction)
    try:
        pipeline_skw_final = pd.read_csv("SKW Final.csv")
        notebook_skw_final = pd.read_csv(notebook_dir / "SKW Final.csv")
        if not compare_dataframes(pipeline_skw_final, notebook_skw_final, "SKW Final.csv"):
            all_tests_passed = False
    except Exception as e:
        print(f"❌ Error comparing SKW Final.csv: {e}")
        all_tests_passed = False
    
    # 2. Compare SKW_final_w_Pd.csv (after VR prediction)
    try:
        pipeline_skw_pd = pd.read_csv("SKW_final_w_Pd.csv")
        notebook_skw_pd = pd.read_csv(notebook_dir / "SKW_final_w_Pd.csv")
        if not compare_dataframes(pipeline_skw_pd, notebook_skw_pd, "SKW_final_w_Pd.csv"):
            all_tests_passed = False
    except Exception as e:
        print(f"❌ Error comparing SKW_final_w_Pd.csv: {e}")
        all_tests_passed = False
    
    # 3. Compare df_all.csv (30-minute resampled data)
    try:
        pipeline_df_all = pd.read_csv("df_all.csv")
        notebook_df_all = pd.read_csv(notebook_dir / "df_all.csv")
        if not compare_dataframes(pipeline_df_all, notebook_df_all, "df_all.csv"):
            all_tests_passed = False
    except Exception as e:
        print(f"❌ Error comparing df_all.csv: {e}")
        all_tests_passed = False
    
    # 4. Compare slopes_df_30menit.csv
    try:
        pipeline_slopes = pd.read_csv("slopes_df_30menit.csv")
        notebook_slopes = pd.read_csv(notebook_dir / "slopes_df_30menit.csv")
        if not compare_dataframes(pipeline_slopes, notebook_slopes, "slopes_df_30menit.csv"):
            all_tests_passed = False
    except Exception as e:
        print(f"❌ Error comparing slopes_df_30menit.csv: {e}")
        all_tests_passed = False
    
    # 5. Compare X_predict_30menit.csv (slope features for prediction)
    try:
        pipeline_x = pd.read_csv("X_predict_30menit.csv")
        notebook_x = pd.read_csv(notebook_dir / "X_predict_30menit.csv")
        if not compare_dataframes(pipeline_x, notebook_x, "X_predict_30menit.csv"):
            all_tests_passed = False
    except Exception as e:
        print(f"❌ Error comparing X_predict_30menit.csv: {e}")
        all_tests_passed = False
    
    # 6. Compare prediction_results_30menit.csv (final failure predictions)
    try:
        pipeline_pred = pd.read_csv("prediction_results_30menit.csv")
        # Try to find notebook predictions file
        notebook_pred_files = list(notebook_dir.glob("*prediction*30*.csv"))
        if notebook_pred_files:
            notebook_pred = pd.read_csv(notebook_pred_files[0])
            if not compare_dataframes(pipeline_pred, notebook_pred, "prediction_results_30menit.csv"):
                all_tests_passed = False
        else:
            print(f"⚠️  No notebook prediction file found to compare")
    except Exception as e:
        print(f"❌ Error comparing predictions: {e}")
        all_tests_passed = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED! Pipeline output matches notebook exactly.")
    else:
        print("❌ SOME TESTS FAILED! There are differences between pipeline and notebook.")
    
    return all_tests_passed


if __name__ == "__main__":
    success = test_skw02()
    sys.exit(0 if success else 1)
