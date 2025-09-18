#!/usr/bin/env python3
"""
Main entry point for the Well Analysis Pipeline.

This script demonstrates how to use the WellAnalysisPipeline to process well data
and generate predictions using pre-trained models.
"""
import argparse
import logging
from pathlib import Path
import sys

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

from src.pipeline import WellAnalysisPipeline
from src.utils import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Well Analysis Pipeline')
    
    parser.add_argument(
        '--well-name', 
        type=str, 
        required=True,
        help='Name of the well (used for input/output file naming)'
    )
    
    parser.add_argument(
        '--input-file', 
        type=str,
        help='Path to the input CSV file. If not provided, looks in data/input/{well_name}.csv'
    )
    
    parser.add_argument(
        '--model', 
        type=str,
        choices=['all', 'discharge_pressure', 'virtual_rate', 'slope', 'failure_prediction'],
        default='all',
        help='Which model(s) to run (default: all)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory. Defaults to data/output under the project root.'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the well analysis pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    log_file = Path('logs') / f'{args.well_name}_analysis.log'
    setup_logging(log_file=log_file, level=log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting analysis for well: {args.well_name}")
    
    try:
        # Initialize the pipeline with optional custom output directory
        output_dir = Path(args.output_dir) if args.output_dir else None
        pipeline = WellAnalysisPipeline(args.well_name, output_dir=output_dir)
        
        # Determine input file path
        input_file = args.input_file
        if input_file is None:
            input_file = Path('data') / 'input' / f'{args.well_name}.csv'
        
        # Load and preprocess data
        pipeline.load_data(input_file)
        
        # Run the specified analysis
        if args.model == 'all':
            # Run full analysis pipeline
            results = pipeline.run_full_analysis()
        else:
            # Run a specific model
            data = pipeline.preprocess_data(args.model)
            results = {args.model: pipeline.predict(args.model, data)}
        
        # Generate and save plots
        pipeline.plot_results(results)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
