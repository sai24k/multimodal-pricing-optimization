# Implementation Plan

- [x] 0. Confirm baseline performance and prepare data



















  - Run existing lgb_sbert.py to confirm 55% SMAPE baseline
  - Verify all required data files exist (train_fold.csv, train_sbert.npy, train_tfidf.npz)
  - Print "Baseline SMAPE: 55.XX%" for tracking
  - _Requirements: 6.1_

- [x] 1. Implement Step 1 - Sparse TF-IDF + SBERT fusion (55% → 35%)













  - [x] 1.1 Create step1_sparse_fusion.py script











    - Load train_tfidf.npz (sparse) and train_sbert.npy (dense)
    - Use scipy.sparse.hstack to combine features efficiently
    - Train LightGBM on sparse matrix with same parameters as baseline
    - Print "Step 1 - TF-IDF+SBERT SMAPE: X.XX%"
    - Save OOF predictions to student_resource/proc/step1_oof.npy
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Implement Step 2 - Add numerical features (35% → 30%)




  - [x] 2.1 Create step2_numerical.py script


    - Extract IPQ values using regex r'IPQ[:\s]*(\d+)' with .fillna(1)
    - Extract brand names as first word of catalog_content
    - Calculate price_per_unit as price/IPQ ratio
    - Add numerical features to sparse matrix using sp.csr_matrix
    - Train LightGBM and print "Step 2 - +Numerical Features SMAPE: X.XX%"
    - Save OOF predictions to student_resource/proc/step2_oof.npy
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_


- [ ] 3. Implement Step 3 - Add CNN image features (30% → 23%)


  - [ ] 3.1 Install required packages and create step3_cnn.py script


    - Install timm, torch, torchvision, cv2, requests packages
    - Create EfficientNet-B0 model using timm.create_model
    - Implement url2vec function with 5-second timeout and zero fallback
    - Process all image URLs and save features to train_cnn.npy
    - Train LightGBM on CNN features only and print "Step 3 - +CNN Features SMAPE: X.XX%"
    - Save OOF predictions to student_resource/proc/step3_oof.npy
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 4. Implement Step 4 - 3-model ensemble (23% → 19%)
  - [ ] 4.1 Create step4_ensemble.py script
    - Load OOF predictions from ridge_tfidf_oof.npy, step2_oof.npy, step3_oof.npy
    - Stack predictions using np.column_stack to create 75k × 3 matrix
    - Train Ridge meta-model with alpha=1.0 on stacked predictions
    - Generate final ensemble predictions and print "Step 4 - Ensemble SMAPE: X.XX%"
    - Print comprehensive summary: "SMAPE Progress: 55% → 35% → 30% → 23% → 19%"
    - Save meta-model to student_resource/proc/meta_model.pkl
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 5. Implement hyperparameter optimization (optional improvement)
  - [ ] 5.1 Create optuna_optimization.py script
    - Use Optuna to optimize LightGBM parameters (num_leaves, learning_rate, feature_fraction)
    - Optimize Ridge alpha values for both TF-IDF and meta-model stages
    - Experiment with different EfficientNet variants (B0, B1, B2) for CNN features
    - Use 5-fold CV with same random_state=42 for fair comparison
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 6. Generate final test predictions and submission
  - [ ] 6.1 Create generate_submission.py script
    - Apply identical feature extraction pipeline to test data
    - Load test_tfidf.npz, test_sbert.npy and process test images for CNN features
    - Extract same numerical features (IPQ, brand, price_per_unit) from test catalog content
    - Generate predictions using all trained models (Ridge, LightGBM, CNN)
    - Apply trained meta-model to combine test predictions
    - Transform predictions using expm1 and clip above 0.01
    - Generate CSV file matching exact submission format with sample_id mapping
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 7. Validate and document results
  - [ ] 7.1 Create comprehensive SMAPE tracking and validation
    - Verify each step achieves expected SMAPE improvement
    - Document actual vs expected performance at each stage
    - Create final performance summary with progression tracking
    - Validate submission file format and completeness
    - _Requirements: 6.1, 6.2, 6.3, 6.4_