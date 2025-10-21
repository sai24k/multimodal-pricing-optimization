# Requirements Document

## Introduction

This feature implements a genius-level rescue plan to dramatically improve SMAPE score from the current baseline of 55% to under 20%. The approach builds on the working baseline (lgb_sbert.py) and adds strategic improvements through sparse feature fusion, advanced feature engineering, and intelligent ensemble stacking. The multimodal approach that achieved 65% SMAPE will be abandoned in favor of this focused optimization strategy.

## Requirements

### Requirement 1

**User Story:** As a data scientist, I want to add sparse TF-IDF features to the working SBERT baseline, so that I can capture both semantic and statistical text patterns for a 10-point SMAPE improvement.

#### Acceptance Criteria

1. WHEN processing text THEN the system SHALL create high-dimensional TF-IDF features (50k+ dimensions) with character and word n-grams
2. WHEN combining features THEN the system SHALL horizontally stack sparse TF-IDF with dense SBERT using scipy.sparse.hstack
3. WHEN training LightGBM THEN the system SHALL handle the sparse matrix directly without converting to dense
4. WHEN evaluating performance THEN the system SHALL print "Step 1 - TF-IDF+SBERT SMAPE: X.XX%" and achieve target reduction from 55% baseline
5. WHEN completing step THEN the system SHALL save results for progress tracking

### Requirement 2

**User Story:** As a data scientist, I want to extract critical numerical features from text, so that I can capture pricing patterns and product characteristics for a 5-point SMAPE improvement.

#### Acceptance Criteria

1. WHEN processing catalog content THEN the system SHALL extract IPQ values using regex pattern r'IPQ[:\s]*(\d+)'
2. WHEN processing text THEN the system SHALL extract brand names as the first word of catalog content
3. WHEN creating derived features THEN the system SHALL calculate price_per_unit as price/IPQ ratio
4. WHEN evaluating performance THEN the system SHALL print "Step 2 - +Numerical Features SMAPE: X.XX%" and show improvement from Step 1
5. WHEN completing step THEN the system SHALL save numerical features for ensemble use

### Requirement 3

**User Story:** As a data scientist, I want to add lightweight CNN image features, so that visual information contributes to predictions without complex preprocessing for a 5-point SMAPE improvement.

#### Acceptance Criteria

1. WHEN processing images THEN the system SHALL use EfficientNet-B0 with timm for fast feature extraction
2. WHEN downloading images THEN the system SHALL implement on-the-fly processing with 5-second timeout
3. WHEN handling failed downloads THEN the system SHALL use zero vectors (1280-D) as fallback features
4. WHEN evaluating performance THEN the system SHALL print "Step 3 - +CNN Features SMAPE: X.XX%" and show improvement from Step 2
5. WHEN completing step THEN the system SHALL save CNN features and OOF predictions for ensemble

### Requirement 4

**User Story:** As a data scientist, I want to implement a strategic 3-model ensemble, so that I can achieve the final push from 35% to under 20% SMAPE through intelligent model stacking.

#### Acceptance Criteria

1. WHEN creating base models THEN the system SHALL train Ridge (TF-IDF), LightGBM (SBERT+numerical), and CNN models separately
2. WHEN generating out-of-fold predictions THEN the system SHALL create OOF matrices for each base model in log-space
3. WHEN training meta-model THEN the system SHALL use Ridge regression to blend the 3 base model predictions
4. WHEN evaluating performance THEN the system SHALL print "Step 4 - Ensemble SMAPE: X.XX%" and achieve target under 20%
5. WHEN completing ensemble THEN the system SHALL save final model weights for test predictions

### Requirement 5

**User Story:** As a data scientist, I want to implement advanced hyperparameter optimization, so that each model component achieves maximum performance within the ensemble framework.

#### Acceptance Criteria

1. WHEN optimizing LightGBM THEN the system SHALL tune num_leaves, learning_rate, and feature_fraction using Optuna
2. WHEN optimizing Ridge THEN the system SHALL find optimal alpha values for both TF-IDF and meta-model stages
3. WHEN optimizing CNN features THEN the system SHALL experiment with different EfficientNet variants (B0, B1, B2)
4. WHEN evaluating optimization THEN the system SHALL use 5-fold CV with the same random_state=42 as baseline

### Requirement 6

**User Story:** As a data scientist, I want to track SMAPE progress at every step, so that I can monitor the effectiveness of each improvement and provide a comprehensive summary.

#### Acceptance Criteria

1. WHEN starting optimization THEN the system SHALL print "Baseline SMAPE: 55.XX%" from original lgb_sbert.py
2. WHEN completing each step THEN the system SHALL print current SMAPE with step description
3. WHEN finishing all steps THEN the system SHALL print comprehensive summary showing SMAPE progression
4. WHEN generating summary THEN the system SHALL include format: "SMAPE Progress: 55% → Step1% → Step2% → Step3% → Final%"

### Requirement 7

**User Story:** As a data scientist, I want to generate the final submission with proper test-time processing, so that the sub-20% SMAPE performance translates to the leaderboard.

#### Acceptance Criteria

1. WHEN processing test data THEN the system SHALL apply identical feature extraction pipeline as training
2. WHEN making ensemble predictions THEN the system SHALL combine all 3 models using optimized meta-model weights
3. WHEN generating final output THEN the system SHALL apply expm1 transformation and clip predictions above 0.01
4. WHEN creating submission THEN the system SHALL match exact CSV format with proper sample_id mapping