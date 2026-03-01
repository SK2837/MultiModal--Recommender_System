#!/usr/bin/env python3
"""
Quick Start Script
Runs a simplified version of the pipeline for quick testing
"""
import sys

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          MULTI-MODAL PERSONALIZED RECOMMENDATION SYSTEM                      ║
║                     Quick Start Guide                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script will guide you through running the recommendation system.

STEPS:
1. Install dependencies
2. Download and prepare data
3. Extract multi-modal features (text + images)
4. Train neural collaborative filtering model
5. Evaluate model performance
6. Launch interactive demo

ESTIMATED TIME: 30-60 minutes (depending on your hardware)

""")

choice = input("Ready to start? (yes/no): ").lower()

if choice not in ['yes', 'y']:
    print("Exiting. Run this script again when you're ready!")
    sys.exit(0)

print("\n" + "=" * 80)
print("STEP 1: Installing Dependencies")
print("=" * 80)

import subprocess
import os

# Install requirements
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    print("✓ Dependencies installed successfully")
except subprocess.CalledProcessError:
    print("❌ Failed to install dependencies. Please run: pip install -r requirements.txt")
    sys.exit(1)

# Set environment
os.environ['PYTHONUNBUFFERED'] = '1'

print("\n" + "=" * 80)
print("STEP 2: Data Acquisition")
print("=" * 80)
print("Downloading MovieLens 100K dataset...")

try:
    from data_acquisition import main as acquire_data
    acquire_data()
    print("✓ Data acquisition complete")
except Exception as e:
    print(f"❌ Data acquisition failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 3: Data Preprocessing")
print("=" * 80)

try:
    from data_preprocessing import main as preprocess_data
    preprocess_data()
    print("✓ Data preprocessing complete")
except Exception as e:
    print(f"❌ Preprocessing failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 4: Feature Extraction")
print("=" * 80)
print("This may take 10-20 minutes depending on your hardware...\n")

# Text features
print("Extracting text features (BERT)...")
try:
    from text_features import main as extract_text
    extract_text(use_tfidf=False)
    print("✓ Text features extracted")
except Exception as e:
    print(f"❌ Text feature extraction failed: {e}")
    print("Falling back to TF-IDF...")
    try:
        extract_text(use_tfidf=True)
        print("✓ Text features extracted (TF-IDF)")
    except:
        print("❌ Text feature extraction failed completely")
        sys.exit(1)

# Visual features
print("\nExtracting visual features (ResNet50)...")
try:
    from visual_features import main as extract_visual
    extract_visual()
    print("✓ Visual features extracted")
except Exception as e:
    print(f"⚠ Visual feature extraction failed: {e}")
    print("Continuing with text features only...")

# Fusion
print("\nFusing multi-modal features...")
try:
    from multimodal_fusion import main as fuse_features
    fuse_features()
    print("✓ Multi-modal fusion complete")
except Exception as e:
    print(f"❌ Feature fusion failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 5: Model Training")
print("=" * 80)
print("Training neural collaborative filtering model...")
print("This may take 15-30 minutes...\n")

# Reduce epochs for quick demo
import config
config.NUM_EPOCHS = 20  # Reduced from 50 for faster demo

try:
    from train import main as train_model
    train_model(model_type="neural_cf")
    print("✓ Model training complete")
except Exception as e:
    print(f"❌ Training failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 6: Model Evaluation")
print("=" * 80)

try:
    from evaluation import main as evaluate_model
    metrics = evaluate_model(model_type="neural_cf")
    print("✓ Model evaluation complete")
except Exception as e:
    print(f"❌ Evaluation failed: {e}")
    sys.exit(1)

print("\n\n")
print("╔══════════════════════════════════════════════════════════════════════════════╗")
print("║                                                                              ║")
print("║                          🎉 SETUP COMPLETE! 🎉                               ║")
print("║                                                                              ║")
print("╚══════════════════════════════════════════════════════════════════════════════╝")

print("\n📊 MODEL PERFORMANCE:")
print(f"  • RMSE: {metrics.get('rmse', 'N/A'):.4f}")
print(f"  • Precision@10: {metrics.get('precision@10', 'N/A'):.4f}")
print(f"  • NDCG@10: {metrics.get('ndcg@10', 'N/A'):.4f}")

print("\n🚀 NEXT STEPS:")
print("  1. Launch the demo: streamlit run app.py")
print("  2. Or run: python pipeline.py --steps demo")

demo_choice = input("\nLaunch demo now? (yes/no): ").lower()

if demo_choice in ['yes', 'y']:
    print("\nStarting demo application...")
    print("The app will open at http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
else:
    print("\nYou can launch the demo later with: streamlit run app.py")
    print("\nThanks for using the Multi-Modal Recommendation System!")
