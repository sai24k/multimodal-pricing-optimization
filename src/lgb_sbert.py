import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

train = pd.read_csv('student_resource/student_resource/proc/train_fold.csv')
X_sbert = np.load('student_resource/student_resource/proc/train_sbert.npy')
y = np.log1p(train.price.values)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros_like(y, dtype=float)

def smape(t, p):
    denom = (np.abs(t) + np.abs(p)) / 2
    return 100 * np.mean(np.abs(t - p) / denom)

for fold, (tr, va) in enumerate(kf.split(X_sbert)):
    model = lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.05,
        num_leaves=127, objective='regression_l1', verbosity=-1
    )
    model.fit(
        X_sbert[tr], y[tr],
        eval_set=[(X_sbert[va], y[va])],
        eval_metric='l1',
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )
    oof[va] = model.predict(X_sbert[va])
    print(f'Fold {fold+1} done')

score = smape(np.expm1(y), np.expm1(oof.clip(0, None)))
print(f'Baseline SMAPE: {score:.2f}%')
