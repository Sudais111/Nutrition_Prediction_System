import os
import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "nutrition_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "nutrition_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "nutrition_encoders.pkl")

# Leave-one-out MAE of the Decision Tree Regressor reported in the prototype notebook
MODEL_UNCERTAINTY_MARGIN = 17.08

# Load nutrition dataset
df = pd.read_csv(DATA_PATH)
for col in ("geography", "indicator", "unit", "status", "survey"):
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["geography", "indicator", "survey", "value"]).reset_index(drop=True)

DEFAULT_YEAR = int(df["year"].mode().iloc[0]) if "year" in df.columns and not df["year"].dropna().empty else 2018
DEFAULT_UNIT = str(df["unit"].mode().iloc[0]) if "unit" in df.columns and not df["unit"].dropna().empty else "%"
DEFAULT_STATUS = str(df["status"].mode().iloc[0]) if "status" in df.columns and not df["status"].dropna().empty else "measured"

# Load trained ML model and categorical encoders
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)


def predict_nutrition_value(geography, indicator, survey):
    """Predict nutrition indicator percentage using the trained Decision Tree model."""
    features = pd.DataFrame(
        [
            {
                "geography": int(encoders["geography"].transform([geography])[0]),
                "indicator": int(encoders["indicator"].transform([indicator])[0]),
                "unit": int(encoders["unit"].transform([DEFAULT_UNIT])[0]),
                "status": int(encoders["status"].transform([DEFAULT_STATUS])[0]),
                "survey": int(encoders["survey"].transform([survey])[0]),
                "year": DEFAULT_YEAR,
            }
        ]
    )
    raw_prediction = float(model.predict(features)[0])
    clamped = max(0.0, min(100.0, raw_prediction))
    return round(clamped, 2)


# Dashboard statistics and chart data
def get_dashboard_data():
    geographies = sorted(df["geography"].dropna().unique().tolist())
    indicators = sorted(df["indicator"].dropna().unique().tolist())
    surveys = sorted(df["survey"].dropna().unique().tolist())

    chart_data = (
        df.groupby("indicator")["value"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
    )

    return {
        "geographies": geographies,
        "indicators": indicators,
        "surveys": surveys,
        "total_records": int(len(df)),
        "total_areas": int(df["geography"].nunique()),
        "total_indicators": int(df["indicator"].nunique()),
        "chart_labels": chart_data.index.tolist(),
        "chart_values": chart_data.values.tolist(),
    }


# Home page
@app.route("/")
def home():
    dashboard_data = get_dashboard_data()
    return render_template("index.html", **dashboard_data)


# Nutrition result (measured lookup with Decision Tree Regressor fallback)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("home"))

    geography = (request.form.get("geography") or "").strip()
    indicator = (request.form.get("indicator") or "").strip()
    survey = (request.form.get("survey") or "").strip()

    dashboard_data = get_dashboard_data()

    if not geography or not indicator or not survey:
        return (
            render_template(
                "index.html",
                error="Please select a geography, nutrition indicator, and survey.",
                selected_geography=geography,
                selected_indicator=indicator,
                selected_survey=survey,
                **dashboard_data,
            ),
            400,
        )

    valid_geographies = set(encoders["geography"].classes_)
    valid_indicators = set(encoders["indicator"].classes_)
    valid_surveys = set(encoders["survey"].classes_)

    if (
        geography not in valid_geographies
        or indicator not in valid_indicators
        or survey not in valid_surveys
    ):
        return (
            render_template(
                "index.html",
                error="Invalid selection provided. Please choose values from the available options.",
                selected_geography=geography,
                selected_indicator=indicator,
                selected_survey=survey,
                **dashboard_data,
            ),
            400,
        )

    # Search for matching measured nutrition record
    matching = df[
        (df["geography"] == geography)
        & (df["indicator"] == indicator)
        & (df["survey"] == survey)
    ]

    if not matching.empty:
        record = matching.iloc[0]
        actual_value = round(float(record["value"]), 2)
        record_year = int(record["year"]) if pd.notna(record.get("year")) else DEFAULT_YEAR
        record_unit = str(record.get("unit") or "%")

        return render_template(
            "index.html",
            result=actual_value,
            result_unit=record_unit,
            result_type="measured",
            result_year=record_year,
            selected_geography=geography,
            selected_indicator=indicator,
            selected_survey=survey,
            **dashboard_data,
        )

    # No measured record exists: generate a clearly tagged ML prediction
    predicted_value = predict_nutrition_value(geography, indicator, survey)
    range_low = round(max(0.0, predicted_value - MODEL_UNCERTAINTY_MARGIN), 1)
    range_high = round(min(100.0, predicted_value + MODEL_UNCERTAINTY_MARGIN), 1)

    return render_template(
        "index.html",
        result=predicted_value,
        result_unit=DEFAULT_UNIT,
        result_type="modelled",
        result_year=DEFAULT_YEAR,
        range_low=range_low,
        range_high=range_high,
        selected_geography=geography,
        selected_indicator=indicator,
        selected_survey=survey,
        **dashboard_data,
    )


# Run Flask application
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
    app.run(debug=debug_mode)

