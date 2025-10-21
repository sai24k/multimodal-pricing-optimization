#!/usr/bin/env python3
"""
Example script for generating predictions using the multimodal pricing optimization system.

This script demonstrates how to use the prediction pipeline to process test data
and generate submission files.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from prediction.prediction_pipeline import PredictionPipeline, create_default_config
from prediction.test_processor import load_test_data
from prediction.submission_generator import validate_submission_format


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('prediction.log')
        ]
    )


def main():
    """Main prediction generation function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting multimodal pricing optimization prediction generation")
    
    # Configuration
    config = create_default_config()
    
    # Update paths to match project structure
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data'
    
    config.update({
        'models': {
            'ensemble_model_path': str(base_dir / 'outputs' / 'ensemble_model.pkl'),
            'base_models': {
                'lightgbm': str(base_dir / 'outputs' / 'lightgbm_model.pkl'),
                'xgboost': str(base_dir / 'outputs' / 'xgboost_model.pkl'),
                'catboost': str(base_dir / 'outputs' / 'catboost_model.pkl'),
                'neural_network': str(base_dir / 'outputs' / 'neural_network_model.pkl')
            }
        },
        'text_processor_path': str(base_dir / 'outputs' / 'text_processor.pkl'),
        'image_processor_path': str(base_dir / 'outputs' / 'image_processor.pkl'),
        'feature_engineer_path': str(base_dir / 'outputs' / 'feature_engineer.pkl'),
        'batch_size': 50,  # Smaller batch size for memory management
        'use_ensemble': True
    })
    
    # File paths
    test_csv_path = str(data_dir / 'raw' / 'test.csv')
    sample_test_csv_path = str(data_dir / 'raw' / 'sample_test.csv')
    image_dir = str(data_dir / 'images')
    output_path = str(base_dir / 'outputs' / 'submission.csv')
    
    # Check if test file exists, use sample if not
    if not Path(test_csv_path).exists():
        if Path(sample_test_csv_path).exists():
            logger.info(f"Test file not found, using sample test file: {sample_test_csv_path}")
            test_csv_path = sample_test_csv_path
        else:
            logger.error("No test file found")
            return False
    
    try:
        # Create prediction pipeline
        pipeline = PredictionPipeline(config)
        
        # Initialize pipeline
        logger.info("Initializing prediction pipeline...")
        if not pipeline.initialize():
            logger.error("Failed to initialize pipeline")
            return False
        
        # Run prediction pipeline
        logger.info("Running prediction pipeline...")
        success = pipeline.run_prediction_pipeline(
            test_csv_path=test_csv_path,
            output_path=output_path,
            image_dir=image_dir if Path(image_dir).exists() else None
        )
        
        if success:
            logger.info(f"Predictions generated successfully: {output_path}")
            
            # Load and display sample predictions
            try:
                import pandas as pd
                submission_df = pd.read_csv(output_path)
                logger.info(f"Submission summary:")
                logger.info(f"  Total predictions: {len(submission_df)}")
                logger.info(f"  Price range: {submission_df['price'].min():.6f} - {submission_df['price'].max():.6f}")
                logger.info(f"  Price mean: {submission_df['price'].mean():.6f}")
                logger.info(f"  Price median: {submission_df['price'].median():.6f}")
                
                # Show first few predictions
                logger.info("First 5 predictions:")
                for _, row in submission_df.head().iterrows():
                    logger.info(f"  {row['sample_id']}: {row['price']:.6f}")
                    
            except Exception as e:
                logger.warning(f"Could not display prediction summary: {str(e)}")
            
            return True
        else:
            logger.error("Prediction generation failed")
            return False
            
    except Exception as e:
        logger.error(f"Error in prediction generation: {str(e)}")
        return False


def generate_sample_predictions():
    """Generate predictions for sample test data only."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    base_dir = Path(__file__).parent
    sample_test_path = base_dir / 'data' / 'raw' / 'sample_test.csv'
    output_path = base_dir / 'outputs' / 'sample_submission.csv'
    
    if not sample_test_path.exists():
        logger.error(f"Sample test file not found: {sample_test_path}")
        return False
    
    try:
        # Load sample test data
        test_df = load_test_data(str(sample_test_path))
        logger.info(f"Loaded {len(test_df)} sample test records")
        
        # Create simple predictions (placeholder)
        # In a real scenario, this would use the trained models
        import numpy as np
        np.random.seed(42)
        
        # Generate random predictions in a reasonable price range
        predictions = np.random.lognormal(mean=3.0, sigma=1.0, size=len(test_df))
        predictions = np.clip(predictions, 0.1, 100.0)  # Reasonable price range
        
        # Create submission dataframe
        import pandas as pd
        submission_df = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'price': predictions
        })
        
        # Save submission
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission_df.to_csv(output_path, index=False)
        
        # Validate submission
        if validate_submission_format(str(output_path), test_df['sample_id'].tolist()):
            logger.info(f"Sample predictions generated successfully: {output_path}")
            return True
        else:
            logger.error("Sample submission validation failed")
            return False
            
    except Exception as e:
        logger.error(f"Error generating sample predictions: {str(e)}")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate predictions for multimodal pricing optimization')
    parser.add_argument('--sample-only', action='store_true', 
                       help='Generate predictions for sample data only (no models required)')
    
    args = parser.parse_args()
    
    if args.sample_only:
        success = generate_sample_predictions()
    else:
        success = main()
    
    sys.exit(0 if success else 1)