"""Simple ML training utilities for classification/regression tasks.

This module now trains a classifier to predict the boolean `periodic` label
by default. The training function is intentionally small and returns the
trained model plus a stats dict (accuracy and basic sizes).
"""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    return df


def _prepare_xy(df: pd.DataFrame, target: str = 'Periodic'):
    """Prepare feature matrix X and target y using canonical features.

    Returns (X, y, feature_cols)
    """
    feature_cols = ['alpha', 'beta', 'delta', 'gamma', 'omega']

    # accept lowercase 'periodic' by copying to the canonical name if present
    if target not in df.columns and 'periodic' in df.columns:
        df = df.copy()
        df[target] = df['periodic']

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[feature_cols].astype(float)

    raw_y = df[target]
    if pd.api.types.is_bool_dtype(raw_y):
        y = raw_y.astype(int)
    elif pd.api.types.is_numeric_dtype(raw_y):
        y = (raw_y != 0).astype(int)
    else:
        y = raw_y.map({True: 1, False: 0, 'True': 1, 'False': 0})
        if y.isnull().any():
            y_num = pd.to_numeric(raw_y, errors='coerce')
            y = (y_num.fillna(0) != 0).astype(int)

    return X, y, feature_cols


def _prepare_xy_regression(df: pd.DataFrame, target: str = 'lyapunov'):
    """Prepare feature matrix X and continuous target y for regression.

    All features and target are cast to float; rows with NaN in target are dropped.
    """
    feature_cols = ['alpha', 'beta', 'delta', 'gamma', 'omega']

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    # drop rows with NaN targets
    df_clean = df.dropna(subset=[target])
    if df_clean.empty:
        raise ValueError("No rows remaining after dropping NaN targets for regression")

    X = df_clean[feature_cols].astype(float)
    y = df_clean[target].astype(float)
    return X, y, feature_cols


def train_rf_model(df: pd.DataFrame, target: str = 'Periodic') -> Tuple[RandomForestClassifier, dict]:
    """Train a RandomForest classifier on the canonical features.

    Returns (model, stats) where stats includes accuracy and metadata.
    """
    X, y, feature_cols = _prepare_xy(df, target=target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    stats = {
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'n_samples': int(len(df)),
        'features': feature_cols,
        'target': target,
        'model_type': 'random_forest',
    }
    return model, stats


def train_mlp_model(df: pd.DataFrame, target: str = 'Periodic', hidden_layer_sizes=(100,), max_iter: int = 300) -> Tuple[MLPClassifier, dict]:
    """Train an MLP classifier on the same canonical features.

    Args:
        hidden_layer_sizes: tuple defining MLP hidden layers (default (100,)).
        max_iter: maximum iterations for the MLP solver.
    """
    X, y, feature_cols = _prepare_xy(df, target=target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    stats = {
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'n_samples': int(len(df)),
        'features': feature_cols,
        'target': target,
        'model_type': 'mlp',
        'mlp_params': {
            'hidden_layer_sizes': hidden_layer_sizes,
            'max_iter': max_iter,
        }
    }
    return model, stats


def train_rf_regressor(df: pd.DataFrame, target: str = 'lyapunov') -> Tuple[RandomForestRegressor, dict]:
    """Train a RandomForest regressor to predict a continuous target (default: lyapunov).

    Returns (model, stats) where stats includes mse and r2.
    """
    X, y, feature_cols = _prepare_xy_regression(df, target=target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestRegressor(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    stats = {
        'mse': mse,
        'r2': r2,
        'n_samples': int(len(y)),
        'features': feature_cols,
        'target': target,
        'model_type': 'random_forest_regressor',
        'n_estimators': 200,
    }
    return model, stats


# keep a backwards-compatible name pointing to the Random Forest trainer
def train_model(*args, **kwargs):
    """Compatibility wrapper: alias for `train_rf_model`.

    Prefer calling `train_rf_model` or `train_mlp_model` explicitly.
    """
    return train_rf_model(*args, **kwargs)
