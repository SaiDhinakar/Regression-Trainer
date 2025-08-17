import os
from flask import Flask, render_template, request, redirect, url_for
from pipeline import train_regression_model
from utils.data_plotting import data_preview

app = Flask(__name__, template_folder="templates", static_folder="static")


# Folder to save uploaded files and outputs
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed file types
ALLOWED_EXTENSIONS = {"csv"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "data_preview_path": None,
        "plot_path": None,
        "mse": None,
        "mae": None,
        "r2": None,
        "model_path": None,
        "show_feature_target_form": False,
        "show_model_form": False,
        "uploaded_file": None,
        "columns": [],
        "selected_feature": None,
        "selected_target": None,
    }
    import pandas as pd

    if request.method == "POST":
        # Step 1: Upload CSV
        file = request.files.get("csv_file")
        if file and file.filename != "":
            if not allowed_file(file.filename):
                return render_template(
                    "index.html", error="Invalid file type", **context
                )
            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            context["uploaded_file"] = filename
            context["columns"] = list(df.columns)
            context["show_feature_target_form"] = True
            return render_template("index.html", **context)
        # Step 2: Select feature/target, show preview
        uploaded_file = request.form.get("uploaded_file")
        feature_col = request.form.get("feature_col")
        target_col = request.form.get("target_col")
        if (
            uploaded_file
            and feature_col
            and target_col
            and not request.form.get("model_name")
        ):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file)
            df = pd.read_csv(filepath)
            X = (
                df[[feature_col]]
                if feature_col in df.columns
                else df.iloc[:, [int(feature_col)]]
            )
            y = (
                df[target_col]
                if target_col in df.columns
                else df.iloc[:, int(target_col)]
            )

            data_preview(
                X,
                y,
                features=[feature_col],
                target=y.name,
                output_dir=app.config["UPLOAD_FOLDER"],
            )
            context["data_preview_path"] = os.path.join("uploads", "data_preview.png")
            context["show_model_form"] = True
            context["uploaded_file"] = uploaded_file
            context["selected_feature"] = feature_col
            context["selected_target"] = target_col
            return render_template("index.html", **context)
        # Step 3: Model selection and training
        if (
            uploaded_file
            and feature_col
            and target_col
            and request.form.get("model_name")
        ):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file)
            model_name = request.form.get("model_name", "Linear Regression")
            test_size = float(request.form.get("test_size", 0.2))
            estimators = int(request.form.get("estimators", 100))

            def parse_col(val):
                if val == "" or val is None:
                    return 0
                if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
                    return int(val)
                if "," in val:
                    items = [v.strip() for v in val.split(",")]
                    return [int(v) if v.isdigit() else v for v in items]
                return val

            parsed_feature = parse_col(feature_col)
            parsed_target = parse_col(target_col)
            result = train_regression_model(
                filepath,
                model_name=model_name,
                feature_col=parsed_feature,
                target_col=parsed_target,
                test_size=test_size,
                estimators=estimators,
                output_dir=app.config["UPLOAD_FOLDER"],
            )
            context.update(
                {
                    "data_preview_path": "uploads/data_preview.png",
                    "plot_path": os.path.join(
                        "uploads", os.path.basename(result["plot_path"])
                    ),
                    "mse": result["mse"],
                    "mae": result["mae"],
                    "r2": result["r2"],
                    "model_path": os.path.join(
                        "uploads", os.path.basename(result["model_path"])
                    ),
                    "show_model_form": True,
                    "uploaded_file": uploaded_file,
                    "selected_feature": feature_col,
                    "selected_target": target_col,
                }
            )
            return render_template("index.html", **context)
    return render_template("index.html", **context)


# Download model endpoint
@app.route("/download_model")
def download_model():
    path = request.args.get("path")
    if not path:
        return "File not found", 404
    abs_path = (
        os.path.join(app.root_path, "static", path) if not os.path.isabs(path) else path
    )
    if not os.path.exists(abs_path):
        return "File not found", 404
    from flask import send_file

    return send_file(abs_path, as_attachment=True)


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
