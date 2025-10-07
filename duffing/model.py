"""Simple ML training utilities to predict gamma from features."""
from typing import Tuple
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    return df


def train_model(df: pd.DataFrame, target='label_gamma') -> Tuple[RandomForestRegressor, dict]:
    X = df.drop(columns=[target])
    # drop non-numeric or label columns
    X = X.select_dtypes('number')
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, {'mse': mse}
