# Train model on provided CSV and print MSE + diagnostics
import sys, os
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from duffing.model import train_mlp_model, load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, confusion_matrix

CSV = os.path.join('data', '01122025_dataset.csv')
print('Loading CSV:', CSV)
df = load_dataset(CSV)
print('Dataset shape:', df.shape)
print('Columns:', df.columns.tolist())

# Train classifier to predict `periodic` label
if 'periodic' not in df.columns:
    raise SystemExit("'periodic' column not found in dataset")

model, stats = train_mlp_model(df, target='periodic', max_iter=1000)
print('Trained model. Accuracy:', stats.get('accuracy'))

# Compute precision and AUC on the same test split used during training
features = stats.get('features', ['alpha', 'beta', 'delta', 'gamma', 'omega'])
X_all = df[features].astype(float)
# prepare y as 0/1
raw_y_all = df['periodic']
if pd.api.types.is_bool_dtype(raw_y_all):
    y_all = raw_y_all.astype(int)
elif pd.api.types.is_numeric_dtype(raw_y_all):
    y_all = (raw_y_all != 0).astype(int)
else:
    y_mapped_all = raw_y_all.map({True: 1, False: 0, 'True': 1, 'False': 0})
    y_all = y_mapped_all.fillna(0).astype(int)

# split with same random_state/test_size as training functions
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)

# predictions and metrics
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
print('Confusion matrix:')
print(conf_mat)
stats['confusion_matrix'] = conf_mat.tolist()
prec = float(precision_score(y_test, y_pred, zero_division=0))
auc = None
try:
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        # some classifiers provide decision_function instead
        probs = model.decision_function(X_test)
    # roc_auc_score requires both classes present
    if len(set(y_test.tolist())) >= 2:
        auc = float(roc_auc_score(y_test, probs))
except Exception:
    auc = None

print('Precision:', prec)
print('AUC:', auc)

# add metrics to stats for callers
stats['precision'] = prec
stats['auc'] = auc

# show a few predictions — construct feature matrix with same columns used for training
features = stats.get('features', ['alpha', 'beta', 'delta', 'gamma', 'omega'])
try:
    X_pred = df[features].astype(float).iloc[:10]
except Exception:
    # last-resort: select numeric columns and hope ordering matches (not recommended)
    X_pred = df.select_dtypes('number').iloc[:10]

# coerce true labels to 0/1 for display if present
if 'periodic' in df.columns:
    raw_y = df['periodic'].iloc[:10]
    if pd.api.types.is_bool_dtype(raw_y):
        y_true = raw_y.astype(int).tolist()
    elif pd.api.types.is_numeric_dtype(raw_y):
        y_true = (raw_y != 0).astype(int).tolist()
    else:
        y_mapped = raw_y.map({True: 1, False: 0, 'True': 1, 'False': 0})
        y_true = y_mapped.fillna(0).astype(int).tolist()
else:
    y_true = None

pred = model.predict(X_pred)
if y_true is not None:
    print('First 10 true y:', y_true)
print('First 10 preds :', [int(x) for x in pred.tolist()])
