#!/usr/bin/env python3
"""Compare pipeline outputs with notebook outputs for SKW-02, SKW-33, and SKW-35"""

import pandas as pd
from pathlib import Path
import numpy as np

def compare_csv_files(pipeline_path, notebook_path, file_desc):
    """Compare two CSV files and report differences"""
    try:
        pl_df = pd.read_csv(pipeline_path)
        nb_df = pd.read_csv(notebook_path)
        
        print(f"\n{'='*60}")
        print(f"📄 {file_desc}")
        print('='*60)
        print(f"Pipeline: {pipeline_path}")
        print(f"Notebook: {notebook_path}")
        
        # Shape comparison
        print(f"\n📊 Shape:")
        print(f"  Pipeline: {pl_df.shape}")
        print(f"  Notebook: {nb_df.shape}")
        
        if pl_df.shape != nb_df.shape:
            print(f"  ⚠️  Shape mismatch!")
            return False
        
        # Column comparison
        pl_cols = set(pl_df.columns)
        nb_cols = set(nb_df.columns)
        
        if pl_cols != nb_cols:
            print(f"\n⚠️  Column mismatch!")
            print(f"  Pipeline only: {pl_cols - nb_cols}")
            print(f"  Notebook only: {nb_cols - pl_cols}")
            return False
        
        # Data comparison
        if pl_df.equals(nb_df):
            print(f"\n✅ PERFECT MATCH! Files are identical.")
            return True
        
        # Find differences
        print(f"\n⚠️  Data differences found")
        
        # Try to identify which columns differ
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
            print(f"  Columns with differences: {diff_cols[:10]}")
            
            # Show sample differences
            for col in diff_cols[:3]:
                mask = pl_df[col] != nb_df[col]
                if mask.any():
                    diff_count = mask.sum()
                    print(f"\n  Column '{col}': {diff_count} differences")
                    sample_idx = mask[mask].index[:3]
                    for idx in sample_idx:
                        print(f"    Row {idx}: Pipeline={pl_df.loc[idx, col]}, Notebook={nb_df.loc[idx, col]}")
        
        return False
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error comparing files: {e}")
        return False

def main():
    wells = ['SKW-02', 'SKW-33', 'SKW-35']
    
    files_to_compare = [
        ('SKW Final.csv', 'SKW Final.csv', 'SKW Final (after DP prediction)'),
        ('SKW_final_w_Pd.csv', 'SKW_final_w_Pd.csv', 'SKW Final with VR'),
        ('df_all.csv', 'df_all.csv', 'df_all (30-min resampled)'),
        ('slopes_df_30menit.csv', 'slopes_df_30menit.csv', 'Slopes (30-min windows)'),
        ('X_predict_30menit.csv', 'X_predict_30menit.csv', 'X_predict (features only)'),
        ('prediction_results_30menit.csv', 'prediction_results_30menit.csv', 'Prediction Results (30-min)'),
        ('result_df_3 jam.csv', 'result_df_3 jam.csv', 'Result (3-hour aggregated)'),
    ]
    
    results = {}
    
    for well in wells:
        print(f"\n\n{'#'*60}")
        print(f"# {well}")
        print(f"{'#'*60}")
        
        well_results = {}
        
        for pipeline_file, notebook_file, desc in files_to_compare:
            # Determine paths
            if pipeline_file in ['SKW Final.csv', 'SKW_final_w_Pd.csv', 'df_all.csv', 
                                'slopes_df_30menit.csv', 'X_predict_30menit.csv', 
                                'prediction_results_30menit.csv', 'result_df_3 jam.csv']:
                # These are saved at project root
                pipeline_path = Path(pipeline_file)
            else:
                pipeline_path = Path(f'test_output/{well}/{pipeline_file}')
            
            notebook_path = Path(f'Test Web/Hasil Bacaan Notebook/{well}/{notebook_file}')
            
            match = compare_csv_files(pipeline_path, notebook_path, f"{well} - {desc}")
            well_results[desc] = match
        
        results[well] = well_results
    
    # Summary
    print(f"\n\n{'#'*60}")
    print("# SUMMARY")
    print(f"{'#'*60}\n")
    
    for well in wells:
        print(f"\n{well}:")
        total = len(results[well])
        matches = sum(1 for v in results[well].values() if v)
        print(f"  ✅ Matches: {matches}/{total}")
        
        if matches < total:
            print(f"  ⚠️  Mismatches:")
            for desc, match in results[well].items():
                if not match:
                    print(f"    - {desc}")
    
    # Overall
    total_files = sum(len(r) for r in results.values())
    total_matches = sum(sum(1 for v in r.values() if v) for r in results.values())
    
    print(f"\n{'='*60}")
    print(f"OVERALL: {total_matches}/{total_files} files match")
    print('='*60)
    
    if total_matches == total_files:
        print("\n🎉🎉🎉 ALL FILES MATCH PERFECTLY! 🎉🎉🎉")
    else:
        print(f"\n⚠️  {total_files - total_matches} files have differences")

if __name__ == '__main__':
    main()
