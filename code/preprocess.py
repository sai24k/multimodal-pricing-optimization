import pandas as pd, numpy as np, re, os, pickle, json
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from scipy import sparse

RAW_DIR  = 'dataset'
PROC_DIR = 'proc'
os.makedirs(PROC_DIR, exist_ok=True)

# ---------- 1. load ----------
train = pd.read_csv(os.path.join(RAW_DIR, 'train.csv'))
test  = pd.read_csv(os.path.join(RAW_DIR, 'test.csv'))
train['price'] = train['price'].astype(float)

# ---------- 2. clean text ----------
def clean_text(s):
    s = re.sub(r'<[^>]+>', ' ', str(s))        # drop HTML
    s = re.sub(r'[^A-Za-z0-9\s]+', ' ', s)     # keep alnum
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s
train['text'] = train.catalog_content.apply(clean_text)
test['text']  = test.catalog_content.apply(clean_text)

# ---------- 3. IPQ ----------
extract_ipq = lambda s: int(re.search(r'IPQ[:\s]*(\d+)', str(s), re.I).group(1)) if re.search(r'IPQ[:\s]*(\d+)', str(s), re.I) else 1
train['ipq'] = train.catalog_content.apply(extract_ipq)
test['ipq']  = test.catalog_content.apply(extract_ipq)

# ---------- 4. TF-IDF ----------
tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1,3), min_df=5, sublinear_tf=True)
train_text_tfidf = tfidf.fit_transform(train.text)
test_text_tfidf  = tfidf.transform(test.text)
pickle.dump(tfidf, open(f'{PROC_DIR}/tfidf.pkl','wb'))
sparse.save_npz(f'{PROC_DIR}/train_tfidf.npz', train_text_tfidf)
sparse.save_npz(f'{PROC_DIR}/test_tfidf.npz',  test_text_tfidf)

# ---------- 5. Sentence-BERT ----------
sent_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-D, Apache-2
train_sbert = sent_model.encode(train.text, batch_size=512, show_progress_bar=True)
test_sbert  = sent_model.encode(test.text,  batch_size=512, show_progress_bar=True)
np.save(f'{PROC_DIR}/train_sbert.npy', train_sbert)
np.save(f'{PROC_DIR}/test_sbert.npy',  test_sbert)

# ---------- 6. folds ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train['fold'] = -1
for f, (tr, va) in enumerate(kf.split(train)):
    train.loc[va, 'fold'] = f
train.to_csv(f'{PROC_DIR}/train_fold.csv', index=False)

# ---------- 7. meta ----------
meta = {'tfidf_shape': train_text_tfidf.shape,
        'sbert_dim': train_sbert.shape[1]}
json.dump(meta, open(f'{PROC_DIR}/meta.json','w'))
print('Pre-processing done – ready for modelling!')