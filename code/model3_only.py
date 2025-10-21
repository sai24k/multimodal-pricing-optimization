#!/usr/bin/env python3
"""
MODEL 3 ONLY - ElasticNet (44.55% SMAPE)
Use only the realistic performing model
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import time

def smape(y_true, y_pred):
    return 100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

def main():
    print("MODEL 3 ONLY - ElasticNet Implementation...")
    start_time = time.time()
    
    # Load data
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    
    # Load SBERT features
    train_sbert = np.load('proc/train_sbert.npy')
    test_sbert = np.load('proc/test_sbert.npy')
    
    # Extract numerical features
    import re
    train['ipq'] = train.catalog_content.str.extract(r'IPQ[:\s]*(\d+)', flags=re.I).fillna(1).astype(int)
    test['ipq'] = test.catalog_content.str.extract(r'IPQ[:\s]*(\d+)', flags=re.I).fillna(1).astype(int)
    
    train['price_per_unit'] = train.price / train.ipq
    test_numerical = test[['ipq']].astype(float).values
    test_numerical = np.column_stack([test_numerical, np.zeros(len(test))])  # dummy price_per_unit
    
    train_numerical = train[['ipq', 'price_per_unit']].astype(float).values
    
    # Combine SBERT + numerical features
    train_combined = np.hstack([train_sbert, train_numerical])
    test_combined = np.hstack([test_sbert, test_numerical])
    
    print(f"Combined features shape: {train_combined.shape}")
    
    # Target
    y_true = train['price'].values
    y_log = np.log1p(y_true)
    
    # Scale features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_combined)
    test_scaled = scaler.transform(test_combined)
    
    print("Training ElasticNet with cross-validation...")
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
        print(f"Fold {fold + 1}/5...")
        
        model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=1000)
        model.fit(train_scaled[tr_idx], y_log[tr_idx])
        
        oof_pred[val_idx] = model.predict(train_scaled[val_idx])
        test_pred += model.predict(test_scaled) / 5
    
    # Calculate SMAPE
    oof_dollars = np.expm1(oof_pred).clip(0.01, None)
    cv_smape = smape(y_true, oof_dollars)
    
    print(f"ElasticNet CV SMAPE: {cv_smape:.2f}%")
    
    # Generate final predictions
    final_prices = np.expm1(test_pred).clip(0.01, None)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test['sample_id'],
        'price': final_prices.round(2)
    })
    
    submission.to_csv('submission_model3.csv', index=False)
    
    total_time = time.time() - start_time
    
    print(f"\nModel 3 submission saved: submission_model3.csv")
    print(f"CV SMAPE: {cv_smape:.2f}%")
    print(f"Price range: {submission['price'].min():.2f} - {submission['price'].max():.2f}")
    print(f"Price mean: {submission['price'].mean():.2f}")
    print(f"Total time: {total_time:.2f}s")
    print("Sample predictions:")
    print(submission.head())
    
    # Sanity check
    sample = pd.read_csv('dataset/sample_test_out.csv')
    print(f"\nSanity check:")
    print(f"Sample mean: {sample['price'].mean():.2f}, Our mean: {submission['price'].mean():.2f}")
    
    return submission

if __name__ == "__main__":
    main()