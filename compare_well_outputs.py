#!/usr/bin/env python3
"""Compare pipeline outputs with notebook outputs for each well separately"""

import pandas as pd
from pathlib import Path
import numpy as np
import sys

def compare_csv_files(pipeline_path, notebook_path, file_desc, well):
    """Compare two CSV files and report differences"""
    try:
        pl_df = pd.read_csv(pipeline_path)
        nb_df = pd.read_csv(notebook_path)
        
        print(f"\n{'='*70}")
        print(f"📄 {well} - {file_desc}")
        print('='*70)
        
        # Shape comparison
        print(f"Shape: Pipeline={pl_df.shape}, Notebook={nb_df.shape}", end='')
        
        if pl_df.shape != nb_df.shape:
            print(f" ⚠️ MISMATCH")
            return False
        print(" ✅")
        
        # Column comparison
        pl_cols = set(pl_df.columns)
        nb_cols = set(nb_df.columns)
        
        if pl_cols != nb_cols:
            print(f"⚠️ Column mismatch!")
            print(f"  Pipeline only: {pl_cols - nb_cols}")
            print(f"  Notebook only: {nb_cols - pl_cols}")
            return False
        
        # Data comparison
        if pl_df.equals(nb_df):
            print(f"✅ PERFECT MATCH - Files are identical")
            return True
        
        # Find differences
        diff_cols = []
        for col in pl_df.columns:
            try:
                if pl_df[col].dtype in ['float64', 'float32']:
                    if not np.allclose(pl_df[col].fillna(0), nb_df[col].fillna(0), rtol=1e-5, atol=1e-8):
                        diff_cols.append(col)
                else:
                    if not pl_df[col].equals(nb_df[col]):
                        diff_cols.append(col)
            except:
                diff_cols.append(col)
        
        if diff_cols:
            print(f"⚠️ Data differences in columns: {diff_cols[:5]}")
            
            # Show sample differences for first differing column
            for col in diff_cols[:2]:
                try:
                    mask = pl_df[col] != nb_df[col]
                    if hasattr(mask, 'any') and mask.any():
                        diff_count = mask.sum()
                        print(f"  '{col}': {diff_count} differences")
                        sample_idx = mask[mask].index[:2]
                        for idx in sample_idx:
                            pl_val = pl_df.loc[idx, col]
                            nb_val = nb_df.loc[idx, col]
                            print(f"    Row {idx}: Pipeline={pl_val}, Notebook={nb_val}")
                except Exception as e:
                    print(f"  '{col}': Error comparing - {e}")
        
        return False
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    wells = ['SKW-02', 'SKW-33', 'SKW-35']
    
    # All files now saved to test_output/{well}/ directory
    well_specific_files = [
        ('SKW Final.csv', 'SKW Final.csv', 'SKW Final (after DP prediction)'),
        ('SKW_final_w_Pd.csv', 'SKW_final_w_Pd.csv', 'SKW Final with VR'),
        ('df_all.csv', 'df_all.csv', 'df_all (30-min resampled)'),
        ('slopes_df_30menit.csv', 'slopes_df_30menit.csv', 'Slopes (30-min windows)'),
        ('X_predict_30menit.csv', 'X_predict_30menit.csv', 'X_predict (features only)'),
        ('{well}_failure_prediction_30min.csv', 'prediction_results_30menit.csv', 'Prediction Results (30-min)'),
        ('result_df_3 jam.csv', 'result_df_3 jam.csv', 'Result (3-hour aggregated)'),
    ]
    
    results = {}
    
    for well in wells:
        print(f"\n\n{'#'*70}")
        print(f"# {well}")
        print(f"{'#'*70}")
        
        well_results = {}
        
        # Compare all files from test_output/{well}/
        for pipeline_file_template, notebook_file, desc in well_specific_files:
            pipeline_file = pipeline_file_template.format(well=well)
            pipeline_path = Path(f'test_output/{well}/{pipeline_file}')
            notebook_path = Path(f'Test Web/Hasil Bacaan Notebook/{well}/{notebook_file}')
            
            match = compare_csv_files(pipeline_path, notebook_path, desc, well)
            well_results[desc] = match
        
        results[well] = well_results
    
    # Summary
    print(f"\n\n{'#'*70}")
    print("# SUMMARY")
    print(f"{'#'*70}\n")
    
    all_match = True
    for well in wells:
        print(f"\n{well}:")
        total = len(results[well])
        matches = sum(1 for v in results[well].values() if v)
        print(f"  ✅ Matches: {matches}/{total}")
        
        if matches < total:
            all_match = False
            print(f"  ⚠️  Mismatches:")
            for desc, match in results[well].items():
                if not match:
                    print(f"    - {desc}")
    
    # Overall
    total_files = sum(len(r) for r in results.values())
    total_matches = sum(sum(1 for v in r.values() if v) for r in results.values())
    
    print(f"\n{'='*70}")
    print(f"OVERALL: {total_matches}/{total_files} files match")
    print('='*70)
    
    if all_match:
        print("\n🎉🎉🎉 ALL FILES MATCH PERFECTLY! 🎉🎉🎉")
        return 0
    else:
        print(f"\n⚠️  {total_files - total_matches} files have differences")
        return 1

if __name__ == '__main__':
    sys.exit(main())
