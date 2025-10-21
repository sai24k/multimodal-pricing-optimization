#!/usr/bin/env python3
"""
Step 6 - Clean Submission (TF-IDF only, no leaked data)
Uses only valid features for competition submission
"""

import numpy as np
import pandas as pd
import pickle
from scipy import sparse
from sklearn.linear_model import Ridge
import time

def main():
    print("Step 6 - Clean Submission Starting...")
    start_time = time.time()
    
    try:
        # Load test data
        test = pd.read_csv('dataset/test.csv')
        print(f"Test data shape: {test.shape}")
        
        # Load test TF-IDF features (clean, no leaked data)
        test_tfidf = sparse.load_npz('proc/test_tfidf.npz')
        print(f"Test TF-IDF shape: {test_tfidf.shape}")
        
        # Load trained Ridge TF-IDF model
        try:
            with open('proc/ridge_tfidf.pkl', 'rb') as f:
                ridge_model = pickle.load(f)
            print("Loaded ridge_tfidf.pkl")
        except:
            # Create Ridge model if not available
            print("Creating Ridge model...")
            from scipy import sparse as sp
            train = pd.read_csv('dataset/train.csv')
            train_tfidf = sp.load_npz('proc/train_tfidf.npz')
            y_log = np.log1p(train['price'].values)
            
            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(train_tfidf, y_log)
        
        # Load meta-model
        with open('proc/meta_clean.pkl', 'rb') as f:
            meta_model = pickle.load(f)
        print("Loaded meta_clean.pkl")
        
        # Pipeline: TF-IDF → Ridge → Meta → expm1 → clip
        print("Generating predictions...")
        
        # Step 1: Ridge prediction on TF-IDF
        ridge_pred = ridge_model.predict(test_tfidf)
        
        # Step 2: Meta-model prediction
        X_meta = ridge_pred.reshape(-1, 1)
        meta_pred = meta_model.predict(X_meta)
        
        # Step 3: Transform to dollars and clip
        final_prices = np.expm1(meta_pred).clip(0.01, None)
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'sample_id': test['sample_id'],
            'price': final_prices.round(2)
        })
        
        # Save submission
        submission.to_csv('submission_clean.csv', index=False)
        
        total_time = time.time() - start_time
        
        # Print results
        print("Clean submission saved: submission_clean.csv")
        print(f"Submission shape: {submission.shape}")
        print(f"Price range: {submission['price'].min():.2f} - {submission['price'].max():.2f}")
        print(f"Price mean: {submission['price'].mean():.2f}")
        print(f"Total execution time: {total_time:.2f}s")
        print("Sample predictions:")
        print(submission.head())
        
        return submission
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()