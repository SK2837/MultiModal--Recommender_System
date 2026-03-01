"""
Master Pipeline Script
Runs the complete multi-modal recommendation system pipeline end-to-end
"""
import argparse
import sys
from pathlib import Path

import config
from utils import setup_logging

logger = setup_logging(log_file=config.LOGS_DIR / "pipeline.log")


def run_data_acquisition():
    """Step 1: Download and acquire data."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: DATA ACQUISITION")
    logger.info("=" * 80)
    
    from data_acquisition import main as acquire_data
    acquire_data()
    
    logger.info("✓ Data acquisition complete\n")


def run_preprocessing():
    """Step 2: Preprocess and prepare data."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: DATA PREPROCESSING")
    logger.info("=" * 80)
    
    from data_preprocessing import main as preprocess_data
    preprocess_data()
    
    logger.info("✓ Data preprocessing complete\n")


def run_text_features():
    """Step 3: Extract text features."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: TEXT FEATURE EXTRACTION")
    logger.info("=" * 80)
    
    from text_features import main as extract_text
    extract_text(use_tfidf=False)
    
    logger.info("✓ Text feature extraction complete\n")


def run_visual_features():
    """Step 4: Extract visual features."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: VISUAL FEATURE EXTRACTION")
    logger.info("=" * 80)
    
    from visual_features import main as extract_visual
    extract_visual()
    
    logger.info("✓ Visual feature extraction complete\n")


def run_multimodal_fusion():
    """Step 5: Fuse multi-modal features."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: MULTI-MODAL FUSION")
    logger.info("=" * 80)
    
    from multimodal_fusion import main as fuse_features
    fuse_features()
    
    logger.info("✓ Multi-modal fusion complete\n")


def run_training(model_type: str = "neural_cf"):
    """Step 6: Train recommendation model."""
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 6: MODEL TRAINING ({model_type.upper()})")
    logger.info("=" * 80)
    
    from train import main as train_model
    train_model(model_type=model_type)
    
    logger.info(f"✓ {model_type} model training complete\n")


def run_evaluation(model_type: str = "neural_cf"):
    """Step 7: Evaluate model."""
    logger.info("\n" + "=" * 80)
    logger.info(f"STEP 7: MODEL EVALUATION ({model_type.upper()})")
    logger.info("=" * 80)
    
    from evaluation import main as evaluate_model
    evaluate_model(model_type=model_type)
    
    logger.info(f"✓ {model_type} model evaluation complete\n")


def run_demo():
    """Step 8: Launch demo application."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: LAUNCHING DEMO APPLICATION")
    logger.info("=" * 80)
    
    import subprocess
    import sys
    
    logger.info("Starting Streamlit app...")
    logger.info("The app will open in your browser at http://localhost:8501")
    logger.info("Press Ctrl+C to stop the demo")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


def main():
    """Run the complete pipeline."""
    parser = argparse.ArgumentParser(description="Multi-Modal Recommendation System Pipeline")
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['data', 'preprocess', 'text', 'visual', 'fusion', 'train', 'eval', 'demo', 'all'],
        default=['all'],
        help='Pipeline steps to run'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='neural_cf',
        choices=['neural_cf', 'two_tower'],
        help='Model type to train/evaluate'
    )
    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='Skip feature extraction (use existing features)'
    )
    
    args = parser.parse_args()
    
    # Determine which steps to run
    if 'all' in args.steps:
        steps = ['data', 'preprocess', 'text', 'visual', 'fusion', 'train', 'eval']
    else:
        steps = args.steps
    
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + "MULTI-MODAL PERSONALIZED RECOMMENDATION SYSTEM PIPELINE".center(78) + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info(f"\nRunning steps: {', '.join(steps)}")
    logger.info(f"Model type: {args.model}\n")
    
    try:
        # Data acquisition
        if 'data' in steps:
            run_data_acquisition()
        
        # Preprocessing
        if 'preprocess' in steps:
            run_preprocessing()
        
        # Feature extraction
        if not args.skip_features:
            if 'text' in steps:
                run_text_features()
            
            if 'visual' in steps:
                run_visual_features()
            
            if 'fusion' in steps:
                run_multimodal_fusion()
        
        # Training
        if 'train' in steps:
            run_training(model_type=args.model)
        
        # Evaluation
        if 'eval' in steps:
            run_evaluation(model_type=args.model)
        
        # Demo
        if 'demo' in steps:
            run_demo()
        
        logger.info("\n" + "╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 78 + "║")
        logger.info("║" + "PIPELINE COMPLETED SUCCESSFULLY!".center(78) + "║")
        logger.info("║" + " " * 78 + "║")
        logger.info("╚" + "=" * 78 + "╝\n")
        
        logger.info("Next steps:")
        if 'demo' not in steps:
            logger.info("  • Run: python pipeline.py --steps demo")
            logger.info("    to launch the interactive demo")
        logger.info("  • Check logs/ directory for detailed logs")
        logger.info("  • Check checkpoints/ for trained models")
        logger.info("  • Evaluation results saved in logs/evaluation_results_*.json")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
