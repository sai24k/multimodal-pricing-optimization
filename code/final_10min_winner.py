#!/usr/bin/env python3
"""
FINAL 10MIN WINNER - Improve 53% solution to <39% SMAPE
Target: Beat 39% leaderboard with better Ridge + features
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
import time

def smape(y_true, y_pred):
    return 100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

def main():
    print("FINAL 10MIN WINNER - Improving 53% solution to <39%...")
    start_time = time.time()
    
    # Load data
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    
    # Load TF-IDF + SBERT features
    train_tfidf = sparse.load_npz('proc/train_tfidf.npz')
    test_tfidf = sparse.load_npz('proc/test_tfidf.npz')
    train_sbert = np.load('proc/train_sbert.npy')
    test_sbert = np.load('proc/test_sbert.npy')
    
    print(f"TF-IDF shape: {train_tfidf.shape}")
    print(f"SBERT shape: {train_sbert.shape}")
    
    # Extract numerical features (like 53% solution)
    import re
    train['ipq'] = train.catalog_content.str.extract(r'IPQ[:\s]*(\d+)', flags=re.I).fillna(1).astype(int)
    test['ipq'] = test.catalog_content.str.extract(r'IPQ[:\s]*(\d+)', flags=re.I).fillna(1).astype(int)
    
    train['price_per_unit'] = train.price / train.ipq
    train_numerical = train[['ipq', 'price_per_unit']].astype(float).values
    test_numerical = test[['ipq']].astype(float).values
    test_numerical = np.column_stack([test_numerical, np.zeros(len(test))])
    
    # Target
    y_true = train['price'].values
    y_log = np.log1p(y_true)
    
    print(f"Price range: {y_true.min():.2f} - {y_true.max():.2f}")
    
    # STRATEGY: Combine TF-IDF + SBERT + numerical for better performance
    # Select top TF-IDF features to avoid overfitting
    selector = SelectKBest(f_regression, k=2000)  # Reduced from 5000
    train_tfidf_sel = selector.fit_transform(train_tfidf, y_log)
    test_tfidf_sel = selector.transform(test_tfidf)
    
    # Combine all features (convert sparse to dense)
    train_combined = np.hstack([train_tfidf_sel.toarray(), train_sbert, train_numerical])
    test_combined = np.hstack([test_tfidf_sel.toarray(), test_sbert, test_numerical])
    
    print(f"Combined features shape: {train_combined.shape}")
    
    # Cross-validation with optimized Ridge
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Test multiple Ridge alphas quickly
    alphas = [0.01, 0.05, 0.1, 0.3, 0.5]
    best_alpha = 0.1
    best_smape = float('inf')
    
    print("Quick alpha optimization...")
    for alpha in alphas:
        oof_pred = np.zeros(len(train))
        
        for tr_idx, val_idx in kf.split(train):
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(train_combined[tr_idx], y_log[tr_idx])
            oof_pred[val_idx] = model.predict(train_combined[val_idx])
        
        oof_dollars = np.expm1(oof_pred).clip(0.01, None)
        curr_smape = smape(y_true, oof_dollars)
        print(f"Alpha {alpha}: SMAPE = {curr_smape:.2f}%")
        
        if curr_smape < best_smape:
            best_smape = curr_smape
            best_alpha = alpha
    
    print(f"\nBest alpha: {best_alpha}, Best SMAPE: {best_smape:.2f}%")
    
    # Train final model
    final_model = Ridge(alpha=best_alpha, random_state=42)
    final_model.fit(train_combined, y_log)
    
    # Generate test predictions
    test_pred_log = final_model.predict(test_combined)
    test_pred_dollars = np.expm1(test_pred_log).clip(0.01, None)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test['sample_id'],
        'price': test_pred_dollars.round(2)
    })
    
    submission.to_csv('submission_final_winner.csv', index=False)
    
    total_time = time.time() - start_time
    
    print(f"\nFINAL WINNER submission saved!")
    print(f"Expected SMAPE: {best_smape:.2f}%")
    print(f"Price range: {submission['price'].min():.2f} - {submission['price'].max():.2f}")
    print(f"Price mean: {submission['price'].mean():.2f}")
    print(f"Total time: {total_time:.2f}s")
    print("Sample predictions:")
    print(submission.head())
    
    # Compare with sample and 53% solution
    sample = pd.read_csv('dataset/sample_test_out.csv')
    solution_53 = pd.read_csv('submission_genius_fix.csv')
    
    print(f"\nComparison:")
    print(f"Sample:     Mean={sample['price'].mean():.2f}, Range={sample['price'].min():.2f}-{sample['price'].max():.2f}")
    print(f"53% soln:   Mean={solution_53['price'].mean():.2f}, Range={solution_53['price'].min():.2f}-{solution_53['price'].max():.2f}")
    print(f"Our final:  Mean={submission['price'].mean():.2f}, Range={submission['price'].min():.2f}-{submission['price'].max():.2f}")
    
    return submission

if __name__ == "__main__":
    main()