import os
import tempfile
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pipeline import train_regression_model

DATA = os.path.join("tests", "test_data.csv")


def test_linear_regression():
    result = train_regression_model(DATA, model_name="Linear Regression")
    assert set(["mse", "mae", "r2", "plot_path", "model_path"]).issubset(result)
    assert os.path.exists(
        os.path.join("static", "uploads", os.path.basename(result["plot_path"]))
    )
    assert os.path.exists(
        os.path.join("static", "uploads", os.path.basename(result["model_path"]))
    )
    assert isinstance(result["mse"], float)
    assert isinstance(result["mae"], float)
    assert isinstance(result["r2"], float)


def test_polynomial_regression():
    result = train_regression_model(DATA, model_name="Polynomial Regression")
    assert set(["mse", "mae", "r2", "plot_path", "model_path"]).issubset(result)
    assert os.path.exists(
        os.path.join("static", "uploads", os.path.basename(result["plot_path"]))
    )
    assert os.path.exists(
        os.path.join("static", "uploads", os.path.basename(result["model_path"]))
    )


def test_random_forest_regressor():
    result = train_regression_model(
        DATA, model_name="Random Forest Regressor", estimators=10
    )
    assert set(["mse", "mae", "r2", "plot_path", "model_path"]).issubset(result)
    assert os.path.exists(
        os.path.join("static", "uploads", os.path.basename(result["plot_path"]))
    )
    assert os.path.exists(
        os.path.join("static", "uploads", os.path.basename(result["model_path"]))
    )


def test_explicit_target_col():
    # Use index 2 for target
    result = train_regression_model(DATA, target_col="target")
    assert set(["mse", "mae", "r2", "plot_path", "model_path"]).issubset(result)
    assert os.path.exists(
        os.path.join("static", "uploads", os.path.basename(result["plot_path"]))
    )
    assert os.path.exists(
        os.path.join("static", "uploads", os.path.basename(result["model_path"]))
    )


def test_different_test_size():
    result = train_regression_model(DATA, test_size=0.5)
    assert set(["mse", "mae", "r2", "plot_path", "model_path"]).issubset(result)
    assert os.path.exists(
        os.path.join("static", "uploads", os.path.basename(result["plot_path"]))
    )
    assert os.path.exists(
        os.path.join("static", "uploads", os.path.basename(result["model_path"]))
    )


def test_different_estimators():
    result = train_regression_model(
        DATA, model_name="Random Forest Regressor", estimators=5
    )
    assert set(["mse", "mae", "r2", "plot_path", "model_path"]).issubset(result)
    assert os.path.exists(
        os.path.join("static", "uploads", os.path.basename(result["plot_path"]))
    )
    assert os.path.exists(
        os.path.join("static", "uploads", os.path.basename(result["model_path"]))
    )


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        train_regression_model("nonexistent.csv")


def test_empty_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_path = os.path.join(tmpdir, "empty.csv")
        pd.DataFrame().to_csv(empty_path, index=False)
        with pytest.raises(Exception):
            train_regression_model(empty_path)


def test_metrics_values():
    result = train_regression_model(DATA)
    assert isinstance(result["mse"], float)
    assert isinstance(result["mae"], float)
    assert isinstance(result["r2"], float)
