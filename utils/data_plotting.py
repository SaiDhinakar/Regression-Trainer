# data_plotting.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def data_preview(X, y, features, target: str, output_dir=None):
    plt.figure(figsize=(10, 5))
    plt.scatter(X.iloc[:, 0], y, alpha=0.5, color="blue", edgecolor="red")
    # Optionally, fit and plot a line if X is 1D
    if X.shape[1] == 1:
        # Fit a line for preview
        coef = np.polyfit(X.iloc[:, 0], y, 1)
        poly1d_fn = np.poly1d(coef)
        x_sorted = np.sort(X.iloc[:, 0])
        plt.plot(
            x_sorted, poly1d_fn(x_sorted), color="red", linewidth=2, label="Fit Line"
        )
        plt.legend()
    plt.title("Data Preview")
    plt.xlabel(features[0])
    plt.ylabel(target)
    plt.grid()
    if output_dir is None:
        output_dir = os.path.join("static", "uploads")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "data_preview.png")
    plt.savefig(path)
    plt.close()
    return path


def plot_actual_vs_predicted(
    X, y_true, y_pred, model_name, feature_to_plot, target_name, output_dir=None
):

    # Convert to numpy arrays for consistency
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle target_name if None
    target_name = target_name or "Target"

    if feature_to_plot is None:
        # Plot actual vs predicted with 45-degree line
        plt.figure(figsize=(10, 5))
        plt.scatter(
            y_true,
            y_pred,
            alpha=0.5,
            color="blue",
            edgecolor="black",
            label="Predictions",
        )
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="red",
            linewidth=2,
            label="Ideal Fit",
        )
        plt.title(f"Actual vs Predicted ({model_name})")
        plt.xlabel(f"Actual {target_name}")
        plt.ylabel(f"Predicted {target_name}")
        plt.legend()
        plt.grid(True)
        if output_dir is None:
            output_dir = os.path.join("static", "uploads")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(
            output_dir, f"{model_name.replace(' ', '_').lower()}_actual_vs_predicted.png"
        )
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    else:
        if not feature_to_plot in X.columns:
            raise ValueError(f"{feature_to_plot} is not a valid feature")

        plt.figure(figsize=(10, 5))
        plt.scatter(
            X[feature_to_plot],
            y_true,
            alpha=0.5,
            color="blue",
            edgecolor="black",
            label="Actual Values",
        )
        plt.scatter(
            X[feature_to_plot],
            y_pred,
            alpha=0.5,
            color="orange",
            marker="x",
            label="Predicted Values",
        )

        # Draw the predicted trend line by sorting
        sorted_idx = np.argsort(X[feature_to_plot])
        x_sorted = X[feature_to_plot].iloc[sorted_idx]
        y_pred_sorted = y_pred[sorted_idx]
        plt.plot(
            x_sorted, y_pred_sorted, color="red", linewidth=2, label="Predicted Trend"
        )

        plt.title(f"{model_name}: {feature_to_plot} vs. {target_name}")
        plt.xlabel(feature_to_plot)
        plt.ylabel(target_name)
        plt.legend()
        plt.grid(True)

        # Save the plot
        if output_dir is None:
            output_dir = os.path.join("static", "uploads")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(
            output_dir, f"{model_name.replace(' ', '_').lower()}_{feature_to_plot}_vs_{target_name}.png"
        )
        plt.savefig(plot_path)
        plt.close()
        return plot_path
