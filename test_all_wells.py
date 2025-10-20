#!/usr/bin/env python3
"""
Test all wells and compare with notebook results
"""
import pandas as pd
from pathlib import Path
from src.pipeline import WellAnalysisPipeline
import sys

def test_well(well_name: str):
    """Test a single well and compare with notebook"""
    print(f"\n{'='*80}")
    print(f"TESTING: {well_name}")
    print(f"{'='*80}")
    
    # Check if notebook result exists
    notebook_path = Path(f"Test Web/Hasil Bacaan Notebook/{well_name}/result_df_3 jam.csv")
    if not notebook_path.exists():
        print(f"⚠️  Notebook result not found: {notebook_path}")
        return None
    
    # Check if sensor data exists
    sensor_path = Path(f"Test Web/Data Sensor/{well_name}.csv")
    if not sensor_path.exists():
        print(f"⚠️  Sensor data not found: {sensor_path}")
        return None
    
    try:
        # Run pipeline
        print(f"\n📊 Running pipeline for {well_name}...")
        pipeline = WellAnalysisPipeline(well_name, output_dir=f'test_output/{well_name}')
        pipeline.run_full_analysis(input_file=sensor_path)
        
        # Load results
        pipeline_result = pd.read_csv(f'test_output/{well_name}/{well_name}_failure_prediction_30min.csv')
        notebook_result = pd.read_csv(notebook_path)
        
        # Compare
        print(f"\n📈 COMPARISON:")
        print(f"\nNotebook status distribution:")
        nb_dist = notebook_result['Dominant Status'].value_counts()
        print(nb_dist)
        
        print(f"\nPipeline status distribution (30-min):")
        pl_dist = pipeline_result['Status'].value_counts()
        print(pl_dist)
        
        # Check specific issues
        issues = []
        
        # Check for missing statuses
        nb_statuses = set(nb_dist.index)
        pl_statuses = set(pl_dist.index)
        
        missing_in_pipeline = nb_statuses - pl_statuses
        extra_in_pipeline = pl_statuses - nb_statuses
        
        if missing_in_pipeline:
            issues.append(f"❌ Missing in pipeline: {missing_in_pipeline}")
        
        if extra_in_pipeline:
            issues.append(f"⚠️  Extra in pipeline: {extra_in_pipeline}")
        
        # Check counts for each status
        for status in nb_statuses:
            nb_count = nb_dist.get(status, 0)
            # Pipeline has 30-min windows, notebook has 3-hour windows
            # So we need to aggregate pipeline to 3-hour for fair comparison
            # For now, just check if status exists
            if status not in pl_statuses:
                issues.append(f"❌ Status '{status}' exists in notebook ({nb_count}) but NOT in pipeline")
        
        if issues:
            print(f"\n⚠️  ISSUES FOUND:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print(f"\n✅ All statuses present!")
        
        return {
            'well': well_name,
            'notebook_statuses': nb_dist.to_dict(),
            'pipeline_statuses': pl_dist.to_dict(),
            'issues': issues
        }
        
    except Exception as e:
        print(f"❌ Error testing {well_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Test all wells"""
    wells = ['SKW-02', 'SKW-07', 'SKW-14', 'SKW-18', 'SKW-29', 'SKW-30', 'SKW-33', 'SKW-35', 'SKW-36']
    
    results = []
    for well in wells:
        result = test_well(well)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\n{result['well']}:")
        if result['issues']:
            for issue in result['issues']:
                print(f"  {issue}")
        else:
            print(f"  ✅ OK")

if __name__ == '__main__':
    # Test specific well if provided
    if len(sys.argv) > 1:
        well = sys.argv[1]
        test_well(well)
    else:
        main()
