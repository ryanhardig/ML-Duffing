# Train model (regression for lyapunov or classification for periodic) on provided CSV and print metrics
import sys, os
import argparse
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from duffing.model import train_rf_regressor, train_rf_model, load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, precision_score, roc_auc_score
import joblib


def parse_args():
    ap = argparse.ArgumentParser(description='Train RF regressor (lyapunov) or RF classifier (periodic) on a CSV')
    ap.add_argument('--csv', type=str, default=os.path.join('data', '01122025_dataset.csv'), help='Path to CSV file')
    ap.add_argument('--mode', type=str, choices=['lyapunov', 'periodic'], default='lyapunov', help='Training target/mode')
    ap.add_argument('--save-model', type=str, default='', help='Path to save trained model (joblib). Default: rf_model_<mode>.joblib')
    return ap.parse_args()


def main():
    args = parse_args()
    CSV = args.csv
    print('Loading CSV:', CSV)
    df = load_dataset(CSV)
    print('Dataset shape:', df.shape)
    print('Columns:', df.columns.tolist())

    if args.mode == 'lyapunov':
        if 'lyapunov' not in df.columns:
            raise SystemExit("'lyapunov' column not found in dataset")

        model, stats = train_rf_regressor(df, target='lyapunov')
        print('Trained RF regressor. MSE:', stats.get('mse'), 'R2:', stats.get('r2'))

        # Save model
        model_path = args.save_model or 'rf_model_lyapunov.joblib'
        joblib.dump(model, model_path)
        print(f'Saved trained model to {model_path}')

        # Compute MSE/R2 on the same split for verification (should match stats)
        features = stats.get('features', ['alpha', 'beta', 'delta', 'gamma', 'omega'])
        X_all = df[features].astype(float)
        y_all = pd.to_numeric(df['lyapunov'], errors='coerce')

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)
        y_pred = model.predict(X_test)
        mse = float(mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        print('Verification MSE:', mse)
        print('Verification R2 :', r2)

        stats['mse_verification'] = mse
        stats['r2_verification'] = r2

        # show a few predictions — construct feature matrix with same columns used for training
        try:
            X_pred = df[features].astype(float).iloc[:10]
        except Exception:
            # last-resort: select numeric columns and hope ordering matches (not recommended)
            X_pred = df.select_dtypes('number').iloc[:10]

        true_vals = df['lyapunov'].astype(float).iloc[:10].tolist()
        pred = model.predict(X_pred)
        print('First 10 true y:', true_vals)
        print('First 10 preds:', [float(x) for x in pred.tolist()])

    else:  # periodic classification
        if 'periodic' not in df.columns:
            raise SystemExit("'periodic' column not found in dataset")
        model, stats = train_rf_model(df, target='periodic')
        print('Trained RF classifier. Accuracy:', stats.get('accuracy'))

        # Save model
        model_path = args.save_model or 'rf_model_periodic.joblib'
        joblib.dump(model, model_path)
        print(f'Saved trained model to {model_path}')

        features = stats.get('features', ['alpha', 'beta', 'delta', 'gamma', 'omega'])
        X_all = df[features].astype(float)
        raw_y_all = df['periodic']
        if pd.api.types.is_bool_dtype(raw_y_all):
            y_all = raw_y_all.astype(int)
        elif pd.api.types.is_numeric_dtype(raw_y_all):
            y_all = (raw_y_all != 0).astype(int)
        else:
            y_mapped_all = raw_y_all.map({True: 1, False: 0, 'True': 1, 'False': 0})
            y_all = y_mapped_all.fillna(0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        conf_mat = confusion_matrix(y_test, y_pred)
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        auc = None
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                probs = model.decision_function(X_test)
            if len(set(y_test.tolist())) >= 2:
                auc = float(roc_auc_score(y_test, probs))
        except Exception:
            auc = None

        print('Verification Accuracy:', acc)
        print('Confusion matrix:')
        print(conf_mat)
        print('Precision:', prec)
        print('AUC:', auc)

        stats.update({
            'accuracy_verification': acc,
            'confusion_matrix': conf_mat.tolist(),
            'precision': prec,
            'auc': auc,
        })

        # show a few predictions
        try:
            X_pred = df[features].astype(float).iloc[:10]
        except Exception:
            X_pred = df.select_dtypes('number').iloc[:10]
        pred = model.predict(X_pred)
        true_vals = y_all.iloc[:10].tolist()
        print('First 10 true y:', true_vals)
        print('First 10 preds:', [int(x) for x in pred.tolist()])


if __name__ == '__main__':
    main()
