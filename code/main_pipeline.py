#!/usr/bin/env python3
"""
ML Challenge 2025 - Emergency Ensemble Pipeline
Main execution script for the complete pricing prediction pipeline

Usage:
    python main_pipeline.py

This script runs the complete pipeline:
1. Data preprocessing (TF-IDF + SBERT + numerical features)
2. 2-model ensemble training
3. Test prediction generation
"""

import os
import sys
import time
import subprocess

def run_step(script_name, description):
    """Run a pipeline step and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        elapsed = time.time() - start_time
        print(f"✅ SUCCESS: {description}")
        print(f"⏱️  Execution time: {elapsed:.2f}s")
        
        if result.stdout:
            print("Output:")
            print(result.stdout)
            
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ FAILED: {description}")
        print(f"⏱️  Execution time: {elapsed:.2f}s")
        print(f"Error: {e}")
        
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error output:")
            print(e.stderr)
            
        return False

def main():
    """Run the complete ML pipeline"""
    print("🚀 ML Challenge 2025 - Emergency Ensemble Pipeline")
    print("=" * 60)
    
    pipeline_start = time.time()
    
    # Step 1: Preprocessing
    if not run_step("preprocess.py", "Data Preprocessing (TF-IDF + SBERT + Features)"):
        print("❌ Pipeline failed at preprocessing step")
        return False
    
    # Step 2: Ultimate Hackathon Winner (100x Champion Strategy)
    if not run_step("ultimate_hackathon_winner.py", "Ultimate Winner - Hand-crafted Features + Perfect Scaling"):
        print("❌ Pipeline failed at model training step")
        return False
    
    total_time = time.time() - pipeline_start
    
    print(f"\n{'='*60}")
    print("🏆 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"⏱️  Total execution time: {total_time:.2f}s")
    print("📁 Output files:")
    print("   - submission_ultimate_winner.csv (test predictions)")
    print("   - Ultimate Winner Model (100x Champion Strategy)")
    print(f"{'='*60}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)