# Regression Trainer Web App

A modern, interactive web application for training and visualizing regression models on your own datasets. Built with Flask, scikit-learn, and Tailwind CSS.

## Features

- **Stepwise workflow:**
  1. Upload your CSV dataset
  2. Select feature and target columns
  3. Instantly preview your data
  4. Choose and configure a regression model (Linear, Polynomial, Random Forest)
  5. Train, visualize results, and download the trained model
- **Beautiful, responsive UI** using Tailwind CSS
- **Live data and model preview** with interactive plots
- **Downloadable trained models** for later use

## Live Demo

👉 [View the live demo here](https://regression-trainer.onrender.com/)

## Quickstart

1. **Clone the repository:**
   ```sh
   git clone https://github.com/SaiDhinakar/Regression-Trainer.git
   cd regression-trainer
   ```
2. **Install dependencies:**
   ```sh
   uv add requirements.txt
   ```
3. **Run the app:**
   ```sh
   uv run main.py
   ```
4. **Open your browser:**
   Visit [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Folder Structure

```
├── main.py                # Flask app entry point
├── pipeline.py            # Model training logic
├── utils/
│   └── data_plotting.py   # Plotting utilities
├── templates/             # Jinja2 HTML templates (Tailwind CSS)
│   └── chunks/            # Modular template chunks
├── static/                # Static files (css, js, images, models)
│   └── uploads/           # All generated images and models
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md
```

## Example Usage

1. Upload a CSV file with your data.
2. Select the feature and target columns from dropdowns.
3. Preview your data instantly.
4. Choose a regression model and set parameters.
5. Train the model, view metrics and plots, and download the trained model.

## Technologies Used

- Python, Flask
- scikit-learn
- matplotlib
- Tailwind CSS
