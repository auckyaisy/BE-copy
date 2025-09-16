"""
Test script for processing SKW-18 well data.
"""
import sys
from pathlib import Path
import pandas as pd

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import WellAnalysisPipeline
from config.config import INPUT_DIR, OUTPUT_DIR

def test_skw18():
    """Test the pipeline with SKW-18 data."""
    # Input and output file paths
    input_file = INPUT_DIR / 'SKW-18.csv'
    
    # Make sure the input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    try:
        # Initialize and run the pipeline
        print("Initializing pipeline for SKW-18...")
        pipeline = WellAnalysisPipeline('SKW-18')
        
        # Run the full analysis
        print("Running full analysis...")
        results = pipeline.run_full_analysis(input_file)
        
        print("\nAnalysis completed successfully!")
        print("\nResults summary:")
        for model_name, predictions in results.items():
            print(f"- {model_name}: {len(predictions)} predictions")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_skw18()
