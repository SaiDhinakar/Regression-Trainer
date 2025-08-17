import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from utils.data_plotting import plot_actual_vs_predicted


def train_regression_model(
    csv_path,
    model_name="Linear Regression",
    feature_col=0,
    target_col=-1,
    test_size=0.2,
    estimators=100,
    random_state=42,
    output_dir=None,
):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Select features and target
    if isinstance(feature_col, int):
        X = df.iloc[:, [feature_col]]
    elif isinstance(feature_col, list):
        X = (
            df.iloc[:, feature_col]
            if all(isinstance(fc, int) for fc in feature_col)
            else df[feature_col]
        )
    else:
        X = df[[feature_col]]

    if target_col == -1:
        y = df.iloc[:, -1]
    elif isinstance(target_col, int):
        y = df.iloc[:, target_col]
    else:
        y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --- Choose model ---
    if model_name == "Linear Regression":
        model = Pipeline(
            [("scaler", StandardScaler()), ("regressor", LinearRegression())]
        )

    elif model_name == "Polynomial Regression":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=3)),
                ("regressor", LinearRegression()),
            ]
        )

    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor(
            n_estimators=estimators, random_state=random_state
        )

    else:
        raise ValueError("Invalid model choice")

    # Fit model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save model in static/uploads
    if output_dir is None:
        output_dir = os.path.join('static', 'uploads')
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}.pkl")
    joblib.dump(model, model_path)

    # Plot actual vs predicted
    feature_name = X.columns[0] if len(X.columns) == 1 else None
    plot_path = plot_actual_vs_predicted(
        X_test,
        y_test,
        y_pred,
        model_name,
        feature_to_plot=feature_name,
        target_name=y.name,
        output_dir=output_dir
    )

    # Return relative paths for frontend
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "plot_path": os.path.join('uploads', os.path.basename(plot_path)),
        "model_path": os.path.join('uploads', os.path.basename(model_path)),
    }
