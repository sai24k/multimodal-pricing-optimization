#!/usr/bin/env python3
"""
Optimize Clean Model - Find best Ridge alpha for <35% SMAPE
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from scipy import sparse
import time

def smape(y_true, y_pred):
    return 100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

def main():
    print("Optimizing Clean Model for <35% SMAPE...")
    start_time = time.time()
    
    # Load data
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    
    # Load TF-IDF features
    train_tfidf = sparse.load_npz('proc/train_tfidf.npz')
    test_tfidf = sparse.load_npz('proc/test_tfidf.npz')
    
    y_true = train['price'].values
    y_log = np.log1p(y_true)
    
    print(f"Training TF-IDF shape: {train_tfidf.shape}")
    
    # Test different Ridge alphas
    best_alpha = 1.0
    best_smape = float('inf')
    
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("Testing Ridge alphas...")
    for alpha in alphas:
        # Train Ridge model
        ridge = Ridge(alpha=alpha, random_state=42)
        ridge.fit(train_tfidf, y_log)
        
        # Predict on training (for validation)
        pred_log = ridge.predict(train_tfidf)
        pred_dollars = np.expm1(pred_log).clip(0.01, None)
        
        curr_smape = smape(y_true, pred_dollars)
        print(f"Alpha {alpha}: SMAPE = {curr_smape:.2f}%")
        
        if curr_smape < best_smape:
            best_smape = curr_smape
            best_alpha = alpha
    
    print(f"\nBest alpha: {best_alpha}, Best SMAPE: {best_smape:.2f}%")
    
    # Train final model with best alpha
    final_ridge = Ridge(alpha=best_alpha, random_state=42)
    final_ridge.fit(train_tfidf, y_log)
    
    # Generate test predictions
    test_pred_log = final_ridge.predict(test_tfidf)
    test_pred_dollars = np.expm1(test_pred_log).clip(0.01, None)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test['sample_id'],
        'price': test_pred_dollars.round(2)
    })
    
    submission.to_csv('submission_optimized.csv', index=False)
    
    # Save optimized model
    with open('proc/ridge_optimized.pkl', 'wb') as f:
        pickle.dump(final_ridge, f)
    
    total_time = time.time() - start_time
    
    print(f"\nOptimized submission saved: submission_optimized.csv")
    print(f"Final SMAPE: {best_smape:.2f}%")
    print(f"Price range: {submission['price'].min():.2f} - {submission['price'].max():.2f}")
    print(f"Price mean: {submission['price'].mean():.2f}")
    print(f"Total time: {total_time:.2f}s")
    print("Sample predictions:")
    print(submission.head())
    
    return submission

if __name__ == "__main__":
    main()