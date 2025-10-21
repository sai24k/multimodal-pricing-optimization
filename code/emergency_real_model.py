#!/usr/bin/env python3
"""
EMERGENCY REAL MODEL - Proper Ridge Regression
Train actual models on the data, not fake baselines
"""

import numpy as np
import pandas as pd
import pickle
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import time

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def main():
    print("EMERGENCY REAL MODEL - Training Proper Ridge Models...")
    start_time = time.time()
    
    try:
        # Load training data
        train = pd.read_csv('student_resource/dataset/train.csv')
        test = pd.read_csv('student_resource/dataset/test.csv')
        
        print(f"Train shape: {train.shape}")
        print(f"Test shape: {test.shape}")
        
        # Load preprocessed features
        train_tfidf = sparse.load_npz('student_resource/proc/train_tfidf.npz')
        test_tfidf = sparse.load_npz('student_resource/proc/test_tfidf.npz')
        train_sbert = np.load('student_resource/proc/train_sbert.npy')
        test_sbert = np.load('student_resource/proc/test_sbert.npy')
        
        print(f"TF-IDF shape: {train_tfidf.shape}")
        print(f"SBERT shape: {train_sbert.shape}")
        
        # Target variable (log transform)
        y = np.log1p(train['price'].values)
        print(f"Target range: {np.expm1(y).min():.2f} - {np.expm1(y).max():.2f}")
        
        # Model 1: Ridge on TF-IDF
        print("\nTraining Model 1: Ridge on TF-IDF...")
        model1 = Ridge(alpha=1.0, random_state=42)
        model1.fit(train_tfidf, y)
        
        # Cross-validation score
        cv_scores1 = cross_val_score(model1, train_tfidf, y, cv=3, scoring='neg_mean_absolute_error')
        print(f"Model 1 CV MAE: {-cv_scores1.mean():.4f}")
        
        # Model 2: Ridge on SBERT + numerical features
        print("\nTraining Model 2: Ridge on SBERT + numerical...")
        
        # Extract numerical features
        def extract_numerical(df):
            import re
            def clean_text(s):
                s = re.sub(r'<[^>]+>', ' ', str(s))
                s = re.sub(r'[^A-Za-z0-9\s]+', ' ', s)
                s = re.sub(r'\s+', ' ', s).strip().lower()
                return s
            
            df = df.copy()
            df['text'] = df['catalog_content'].apply(clean_text)
            
            # IPQ extraction
            def extract_ipq(s):
                match = re.search(r'IPQ[:\s]*(\d+)', str(s), re.I)
                return int(match.group(1)) if match else 1
            
            df['ipq'] = df['catalog_content'].apply(extract_ipq)
            df['text_length'] = df['text'].str.len()
            df['word_count'] = df['text'].str.split().str.len()
            df['has_numbers'] = df['text'].str.contains(r'\d', na=False).astype(int)
            df['has_price_terms'] = df['text'].str.contains(r'price|cost|dollar|\$', case=False, na=False).astype(int)
            
            return df[['ipq', 'text_length', 'word_count', 'has_numbers', 'has_price_terms']].values
        
        train_numerical = extract_numerical(train)
        test_numerical = extract_numerical(test)
        
        # Combine SBERT + numerical
        train_combined = np.hstack([train_sbert, train_numerical])
        test_combined = np.hstack([test_sbert, test_numerical])
        
        print(f"Combined features shape: {train_combined.shape}")
        
        model2 = Ridge(alpha=1.0, random_state=42)
        model2.fit(train_combined, y)
        
        cv_scores2 = cross_val_score(model2, train_combined, y, cv=3, scoring='neg_mean_absolute_error')
        print(f"Model 2 CV MAE: {-cv_scores2.mean():.4f}")
        
        # Generate out-of-fold predictions for ensemble
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        oof1 = np.zeros(len(train))
        oof2 = np.zeros(len(train))
        
        print("\nGenerating out-of-fold predictions...")
        for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
            # Model 1
            m1 = Ridge(alpha=1.0, random_state=42)
            m1.fit(train_tfidf[tr_idx], y[tr_idx])
            oof1[val_idx] = m1.predict(train_tfidf[val_idx])
            
            # Model 2
            m2 = Ridge(alpha=1.0, random_state=42)
            m2.fit(train_combined[tr_idx], y[tr_idx])
            oof2[val_idx] = m2.predict(train_combined[val_idx])
        
        # Train meta-model
        print("\nTraining meta-model...")
        X_meta = np.column_stack([oof1, oof2])
        meta_model = Ridge(alpha=0.1, random_state=42)
        meta_model.fit(X_meta, y)
        
        # Calculate ensemble SMAPE
        meta_pred = meta_model.predict(X_meta)
        ensemble_smape = smape(np.expm1(y), np.expm1(meta_pred))
        print(f"Ensemble SMAPE: {ensemble_smape:.2f}%")
        
        # Generate test predictions
        print("\nGenerating test predictions...")
        test_pred1 = model1.predict(test_tfidf)
        test_pred2 = model2.predict(test_combined)
        
        X_test_meta = np.column_stack([test_pred1, test_pred2])
        final_pred = meta_model.predict(X_test_meta)
        
        # Transform back to original scale
        final_prices = np.expm1(final_pred)
        final_prices = np.clip(final_prices, 0.01, None)
        
        # Create submission
        submission = pd.DataFrame({
            'sample_id': test['sample_id'],
            'price': final_prices.round(2)
        })
        
        submission.to_csv('test_out_real.csv', index=False)
        
        total_time = time.time() - start_time
        
        print(f"\nReal model submission saved: test_out_real.csv")
        print(f"Ensemble SMAPE: {ensemble_smape:.2f}%")
        print(f"Price range: {submission['price'].min():.2f} - {submission['price'].max():.2f}")
        print(f"Price mean: {submission['price'].mean():.2f}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Sample predictions:")
        print(submission.head())
        
        # Save models
        with open('student_resource/proc/model1_real.pkl', 'wb') as f:
            pickle.dump(model1, f)
        with open('student_resource/proc/model2_real.pkl', 'wb') as f:
            pickle.dump(model2, f)
        with open('student_resource/proc/meta_real.pkl', 'wb') as f:
            pickle.dump(meta_model, f)
        
        return submission
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()