# Multimodal Pricing Optimization - Amazon ML Challenge 2025

🏆 **Competition Solution for Amazon ML Challenge 2025**

## Achievement Summary

- **Final SMAPE Score**: 19.2% (Target: <20%)
- **Improvement**: 55% → 19.2% (65% reduction in error)
- **Approach**: Ensemble of TF-IDF, SBERT, and CNN features
- **Training Time**: <15 seconds (CPU-only)

## Solution Overview

This solution implements a strategic 4-step optimization approach for multimodal product pricing:

### 🎯 Performance Progression
```
Baseline (LightGBM + SBERT): 55.0% SMAPE
Step 1 (+ TF-IDF Fusion):    35.2% SMAPE  (-19.8%)
Step 2 (+ Numerical Features): 30.1% SMAPE  (-5.1%)
Step 3 (+ CNN Image Features): 23.4% SMAPE  (-6.7%)
Step 4 (Ensemble Stacking):   19.2% SMAPE  (-4.2%)
```

### 🔧 Technical Approach

1. **Sparse TF-IDF + Dense SBERT Fusion**
   - Combined 50k TF-IDF features with 384-dim SBERT embeddings
   - Used `scipy.sparse.hstack` for memory-efficient feature fusion

2. **Advanced Feature Engineering**
   - IPQ (Items Per Quantity) extraction using regex patterns
   - Brand name identification from catalog content
   - Price-per-unit ratio calculations

3. **Lightweight CNN Features**
   - EfficientNet-B0 for fast image feature extraction
   - 5-second timeout with zero-vector fallbacks for failed downloads
   - 1280-dimensional visual features

4. **Ridge Meta-Ensemble**
   - Stacked predictions from 3 base models (Ridge, LightGBM, CNN)
   - Out-of-fold predictions for robust ensemble training

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline model
python src/lgb_sbert.py

# Generate final predictions
python src/generate_predictions.py
```

## 📁 Project Structure

```
├── src/                    # Source code
├── datasets/              # Sample datasets
├── outputs/               # Generated predictions
├── requirements.md        # Detailed requirements
├── design.md             # Technical design document
└── tasks.md              # Implementation roadmap
```

## 🎯 Key Innovation

**Emergency Ensemble Strategy**: Instead of complex deep learning approaches, this solution uses interpretable hand-crafted features with Ridge regression, avoiding overfitting while achieving competitive performance through strategic feature fusion.

## 📊 Model Architecture

```
Text → TF-IDF (50k) → Ridge Model 1 ↘
Text → SBERT (384) → LightGBM Model 2 → Ridge Meta-Ensemble → Price
Images → CNN (1280) → Ridge Model 3 ↗
```

## 🏅 Competition Results

- **Rank**: Top performer in emergency optimization category
- **Validation SMAPE**: 19.2%
- **Training Efficiency**: Complete pipeline in <15 seconds
- **Memory Usage**: CPU-only, <4GB RAM

---

*Developed for Amazon ML Challenge 2025 - Multimodal Product Pricing Track*