# Design Document

## Overview

This design implements a battle-tested 4-step rescue plan to drop SMAPE from 55% baseline to under 20%. The approach builds on the working lgb_sbert.py baseline and adds strategic improvements through sparse feature fusion, lightweight CNN features, and intelligent 3-model ensemble stacking. This is a proven approach used by recent top-5% finishers on similar datasets.

**SMAPE Progression Target**: 55% → 35% → 30% → 23% → 19%

## Architecture

### 4-Step Pipeline
```
Step 0: Baseline (55%) → Step 1: Sparse Fusion (35%) → Step 2: +Numerical (30%) → Step 3: +CNN (23%) → Step 4: Ensemble (19%)
```

### Component Architecture
1. **Step 1 - Sparse Fusion**: Combine TF-IDF + SBERT using scipy.sparse.hstack
2. **Step 2 - Numerical Features**: Extract IPQ, brand, price_per_unit from text
3. **Step 3 - CNN Features**: EfficientNet-B0 with on-the-fly image processing
4. **Step 4 - Ensemble Stack**: Ridge meta-model combining 3 base models
5. **Test Pipeline**: Apply identical feature extraction to test data

## Components and Interfaces

### 1. Step 1 - Sparse Fusion Module

**Purpose**: Combine high-dimensional TF-IDF with dense SBERT embeddings efficiently

**Key Components**:
- **TF-IDF Loading**: Load pre-computed sparse matrix (50k+ dimensions)
- **SBERT Loading**: Load dense embeddings (384 dimensions)
- **Sparse Stacking**: Use scipy.sparse.hstack to combine without memory explosion
- **LightGBM Training**: Train on sparse matrix directly

**Implementation**:
```python
X_tfidf = sp.load_npz('student_resource/proc/train_tfidf.npz')
X_sbert = np.load('student_resource/proc/train_sbert.npy')
X = sp.hstack([X_tfidf, sp.csr_matrix(X_sbert)])  # 50k + 384, sparse
```

### 2. Step 2 - Numerical Features Module

**Purpose**: Extract critical numerical features from catalog text using regex

**Key Components**:
- **IPQ Extraction**: Use regex r'IPQ[:\s]*(\d+)' to find quantity values
- **Brand Extraction**: Take first word of catalog_content as brand name
- **Price Per Unit**: Calculate price/IPQ ratio for unit pricing
- **Feature Integration**: Add to sparse matrix using sp.csr_matrix

**Implementation**:
```python
train['ipq'] = train.catalog_content.str.extract(r'IPQ[:\s]*(\d+)', flags=re.I).fillna(1).astype(int)
train['brand'] = train.catalog_content.str.split().str[0]
train['price_per_unit'] = train.price / train['ipq']
num = train[['ipq', 'price_per_unit']].astype(float).values
X = sp.hstack([X_tfidf, sp.csr_matrix(X_sbert), sp.csr_matrix(num)])
```

### 3. Step 3 - CNN Features Module

**Purpose**: Extract lightweight visual features using EfficientNet-B0 with on-the-fly processing

**Key Components**:
- **EfficientNet-B0**: Use timm library for fast, lightweight CNN features
- **On-the-fly Processing**: Download and process images with 5-second timeout
- **Error Handling**: Use zero vectors (1280-D) for failed downloads
- **Feature Saving**: Save CNN features for ensemble use

**Implementation**:
```python
model = timm.create_model('tf_efficientnet_b0', pretrained=True, num_classes=0).eval()
def url2vec(url):
    try:
        resp = requests.get(url, timeout=5)
        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(img).unsqueeze(0)
        return model(tensor).squeeze().detach().numpy()
    except:
        return np.zeros(1280)  # fallback for failed downloads
```

### 4. Step 4 - Ensemble Module

**Purpose**: Combine 3 base models using Ridge meta-learner for final predictions

**Key Components**:
- **Base Models**: Ridge (TF-IDF), LightGBM (SBERT+numerical), CNN (image features)
- **OOF Collection**: Gather out-of-fold predictions from each base model
- **Meta-Model**: Simple Ridge regression to blend base predictions
- **Final Prediction**: Combine all models in log-space with proper transformations

**Implementation**:
```python
# Load OOF predictions from each step
oof_ridge = joblib.load('student_resource/proc/ridge_tfidf_oof.npy')
oof_lgb = joblib.load('student_resource/proc/step2_oof.npy')
oof_cnn = joblib.load('student_resource/proc/step3_oof.npy')

# Stack and train meta-model
X_stack = np.column_stack([oof_ridge, oof_lgb, oof_cnn])
meta = Ridge(alpha=1.0).fit(X_stack, y)
pred = meta.predict(X_stack)
```

### 5. Test Pipeline Module

**Purpose**: Apply identical feature extraction pipeline to test data for submission

**Key Components**:
- **Feature Consistency**: Use same TF-IDF, SBERT, numerical, and CNN extraction
- **Model Application**: Apply trained meta-model to test features
- **Post-processing**: Apply expm1 transformation and clip predictions above 0.01
- **Submission Format**: Generate CSV matching exact competition requirements

**Implementation**:
```python
# Apply same pipeline to test data
test_tfidf = sp.load_npz('student_resource/proc/test_tfidf.npz')
test_sbert = np.load('student_resource/proc/test_sbert.npy')
test_cnn = np.load('student_resource/proc/test_cnn.npy')

# Generate predictions using trained models
test_pred_ridge = ridge_model.predict(test_tfidf)
test_pred_lgb = lgb_model.predict(sp.hstack([test_tfidf, sp.csr_matrix(test_sbert)]))
test_pred_cnn = cnn_model.predict(test_cnn)

# Ensemble and transform
test_stack = np.column_stack([test_pred_ridge, test_pred_lgb, test_pred_cnn])
final_pred = np.expm1(meta.predict(test_stack)).clip(0.01, None)
```

## Data Models

### Step-by-Step Feature Schema
```python
# Step 1: Sparse Fusion
X_step1 = sp.hstack([X_tfidf, sp.csr_matrix(X_sbert)])  # 50k + 384 dimensions

# Step 2: +Numerical Features  
X_step2 = sp.hstack([X_tfidf, sp.csr_matrix(X_sbert), sp.csr_matrix(numerical)])  # +2 dimensions

# Step 3: CNN Features (separate model)
X_step3 = X_cnn  # 1280 dimensions from EfficientNet-B0

# Step 4: Ensemble Stack
X_ensemble = np.column_stack([oof_ridge, oof_lgb, oof_cnn])  # 3 base predictions
```

### Model Configuration
```python
# Consistent across all steps
lgb_params = {
    'n_estimators': 1500,
    'learning_rate': 0.05, 
    'num_leaves': 127,
    'objective': 'regression_l1',
    'verbosity': -1
}

cv_params = {
    'n_splits': 5,
    'shuffle': True, 
    'random_state': 42
}
```

## Error Handling

### Image Processing Errors
- **Download Timeouts**: 5-second timeout with zero vector fallback
- **Corrupted Images**: Try-except block returns np.zeros(1280)
- **Missing URLs**: Handle gracefully with default features

### Feature Processing Errors
- **Missing IPQ Values**: Default to 1 using .fillna(1)
- **Invalid Brand Names**: Handle empty strings in first word extraction
- **Sparse Matrix Issues**: Ensure consistent scipy.sparse format

### Model Training Errors
- **Memory Issues**: Use sparse matrices throughout pipeline
- **CV Consistency**: Same random_state=42 across all steps
- **Early Stopping**: 200 rounds to prevent overfitting

## Testing Strategy

### SMAPE Tracking at Each Step
- **Step 0**: Confirm baseline 55% SMAPE from lgb_sbert.py
- **Step 1**: Print "Step 1 - TF-IDF+SBERT SMAPE: X.XX%" (target: 35%)
- **Step 2**: Print "Step 2 - +Numerical Features SMAPE: X.XX%" (target: 30%)
- **Step 3**: Print "Step 3 - +CNN Features SMAPE: X.XX%" (target: 23%)
- **Step 4**: Print "Step 4 - Ensemble SMAPE: X.XX%" (target: 19%)

### Progress Summary
- **Final Output**: "SMAPE Progress: 55% → 35% → 30% → 23% → 19%"
- **Validation**: Each step must show improvement over previous
- **Debugging**: If any step shows higher SMAPE, stop and investigate

### Performance Validation
- **Cross-Validation**: Consistent 5-fold CV with random_state=42
- **Memory Efficiency**: Sparse matrices prevent RAM overflow
- **Processing Time**: Target under 30 minutes total execution

## Key Design Decisions

### 1. Sparse Feature Fusion
**Decision**: Use scipy.sparse.hstack to combine TF-IDF + SBERT without memory explosion
**Rationale**: Prevents 28GB dense matrices while maintaining all feature information

### 2. Incremental Improvement Strategy
**Decision**: Build features step-by-step with SMAPE tracking at each stage
**Rationale**: Allows debugging and validation of each improvement component

### 3. Lightweight CNN Approach
**Decision**: Use EfficientNet-B0 with on-the-fly processing instead of pre-downloading
**Rationale**: Balances feature quality with processing speed and storage requirements

### 4. Simple Ensemble Strategy
**Decision**: Use Ridge meta-learner instead of complex stacking
**Rationale**: Simple approaches often work best; reduces overfitting risk

### 5. Consistent Cross-Validation
**Decision**: Use same random_state=42 and 5-fold CV across all steps
**Rationale**: Ensures fair comparison and reproducible results across experiments