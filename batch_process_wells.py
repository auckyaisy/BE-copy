#!/usr/bin/env python3
"""
Batch processing script untuk menjalankan pipeline analisis untuk semua sumur
di folder Test Web/Hasil Bacaan Notebook dengan data produksi dari Test Web/Data Produksi
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add parent directory to path for proper imports
base_path = Path(__file__).parent
sys.path.insert(0, str(base_path))

from src.pipeline import WellAnalysisPipeline

def main():
    # Base directories
    base_dir = Path(__file__).parent
    sensor_base = base_dir / "Test Web" / "Hasil Bacaan Notebook"
    prod_base = base_dir / "Test Web" / "Data Produksi"
    output_base = base_dir / "test_output"
    
    # List all well folders
    well_folders = [d for d in sensor_base.iterdir() if d.is_dir() and d.name.startswith("SKW-")]
    well_folders = sorted(well_folders, key=lambda x: x.name)
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING: {len(well_folders)} WELLS")
    print(f"{'='*80}\n")
    
    results = []
    
    for well_folder in well_folders:
        well_name = well_folder.name
        
        print(f"\n{'='*80}")
        print(f"Processing: {well_name}")
        print(f"{'='*80}")
        
        # Paths
        sensor_file = well_folder / "SKW Final.csv"
        prod_file = prod_base / f"{well_name}.csv"
        output_dir = output_base / well_name
        
        # Check if files exist
        if not sensor_file.exists():
            print(f"❌ SKIP: Sensor file not found: {sensor_file}")
            results.append({
                'well': well_name,
                'status': 'FAILED',
                'reason': 'Sensor file not found'
            })
            continue
            
        if not prod_file.exists():
            print(f"❌ SKIP: Production file not found: {prod_file}")
            results.append({
                'well': well_name,
                'status': 'FAILED',
                'reason': 'Production file not found'
            })
            continue
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"📁 Sensor data: {sensor_file}")
            print(f"📁 Production data: {prod_file}")
            print(f"📁 Output dir: {output_dir}")
            
            # Initialize pipeline
            pipeline = WellAnalysisPipeline(
                well_name=well_name,
                output_dir=str(output_dir)
            )
            
            # Set production data path
            pipeline.prod_data_path = prod_file
            
            # Load sensor data
            pipeline.load_data(file_path=sensor_file)
            
            # Run full analysis
            print(f"\n🚀 Running full analysis for {well_name}...")
            start_time = datetime.now()
            
            analysis_results = pipeline.run_full_analysis()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"✅ SUCCESS: {well_name} completed in {duration:.1f}s")
            
            results.append({
                'well': well_name,
                'status': 'SUCCESS',
                'duration_sec': duration,
                'output_dir': str(output_dir)
            })
            
        except Exception as e:
            print(f"❌ ERROR processing {well_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'well': well_name,
                'status': 'ERROR',
                'reason': str(e)
            })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed_count = len(results) - success_count
    
    print(f"\nTotal Wells: {len(results)}")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {failed_count}")
    
    print(f"\n{'Well':<15} {'Status':<10} {'Duration':<12} {'Reason/Output'}")
    print("-" * 80)
    
    for r in results:
        well = r['well']
        status = r['status']
        
        if status == 'SUCCESS':
            duration = f"{r.get('duration_sec', 0):.1f}s"
            info = r.get('output_dir', '')
        else:
            duration = '-'
            info = r.get('reason', 'Unknown error')
        
        print(f"{well:<15} {status:<10} {duration:<12} {info}")
    
    print(f"\n{'='*80}\n")
    
    # Save summary to CSV
    summary_file = output_base / "batch_processing_summary.csv"
    df_summary = pd.DataFrame(results)
    df_summary.to_csv(summary_file, index=False)
    print(f"📄 Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
