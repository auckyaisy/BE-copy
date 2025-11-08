#!/usr/bin/env python3
"""
Quick test to verify Watercut loading and Shut-in/EDP priority fixes.
"""
from pathlib import Path
import subprocess
import sys

wells_to_test = ['SKW-02', 'SKW-07', 'SKW-30', 'SKW-33']

print("=" * 80)
print("Testing Watercut loading and Shut-in/EDP priority fixes")
print("=" * 80)

for well in wells_to_test:
    print(f"\n{'='*80}")
    print(f"Testing {well}")
    print(f"{'='*80}")
    
    sensor_file = Path(f'Test Web/Data Sensor/{well}.csv')
    output_dir = Path(f'test_output/{well}')
    
    if not sensor_file.exists():
        print(f"⚠️  Sensor file not found: {sensor_file}")
        continue
    
    # Run pipeline with INFO logging to see Watercut loading
    cmd = [
        'prod_env/bin/python3', 'main.py',
        '--well-name', well,
        '--input-file', str(sensor_file),
        '--output-dir', str(output_dir),
        '--log-level', 'INFO'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check for Watercut loading messages
    if 'Loaded Watercut data' in result.stderr or 'Loaded Watercut data' in result.stdout:
        print("✅ Watercut data loaded successfully")
    else:
        print("❌ Watercut data NOT loaded - check logs")
    
    # Check for override counts
    if 'Override counts' in result.stderr or 'Override counts' in result.stdout:
        print("✅ Override logic executed")
        # Extract and show the counts
        for line in (result.stderr + result.stdout).split('\n'):
            if 'Override counts' in line or 'Watercut=100%' in line:
                print(f"   {line.strip()}")
    else:
        print("⚠️  No override count logs found")
    
    if result.returncode != 0:
        print(f"❌ Pipeline failed with return code {result.returncode}")
        print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
    else:
        print(f"✅ Pipeline completed successfully")
        
        # Check if output file exists
        output_file = output_dir / 'result_df_3 jam.csv'
        if output_file.exists():
            print(f"✅ Output file created: {output_file}")
        else:
            print(f"❌ Output file missing: {output_file}")

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)
