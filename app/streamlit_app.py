"""Run locally: python -m streamlit run app/streamlit_app.py

Optional explicit saved run: ... -- --run models/final_v1
The UI reads training-derived choices and predicts without fitting any model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import streamlit as st

from fuel_consumption.predict import discover_run, load_verified_model, predict_vehicle
from fuel_consumption.utils import project_root

st.set_page_config(page_title="EPA consumption estimate", page_icon="🚗", layout="centered")
st.title("EPA fuel consumption estimate")
st.write("Compare the estimated combined consumption of a gasoline car or SUV from the US catalogue.")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--run")
args, _ = parser.parse_known_args()


@st.cache_resource
def get_model(run_path: str, metadata_version: int):
    return load_verified_model(run_path)


try:
    run = Path(args.run).resolve() if args.run else discover_run(project_root())
    bundle = get_model(str(run), (run / "run_metadata.json").stat().st_mtime_ns)
except (ValueError, OSError, KeyError):
    st.info("The prediction model is not available yet. Complete a saved experiment before opening this page.")
    st.stop()

if bundle.metadata.get("development_small_dataset", False):
    st.caption("Development sample with fewer than 1,000 configurations. Estimates have limited dataset support.")
else:
    st.caption("Gasoline cars, station wagons and SUVs, model years 2015–2025.")

schema = bundle.interface
choices = schema["categories"]
defaults = schema["numeric_defaults"]
if any(not choices[field] for field in choices):
    st.info("This saved model does not have enough vehicle specifications for the prediction form.")
    st.stop()

with st.form("vehicle_specs"):
    left, right = st.columns(2)
    with left:
        manufacturer = st.selectbox("Manufacturer", choices["manufacturer"])
        years = schema["allowed_years"]
        default_year = int(defaults.get("model_year") or years[-1])
        year = st.selectbox("Model year", years, index=years.index(default_year) if default_year in years else len(years) - 1)
        displacement = st.number_input("Engine displacement (litres)", min_value=0.1, max_value=20.0,
                                       value=max(0.1, min(20.0, float(defaults.get("displacement_l") or 2.0))), step=0.1)
        cylinders = st.number_input("Cylinder count", min_value=1, max_value=32,
                                    value=max(1, min(32, int(round(defaults.get("cylinders") or 4)))), step=1)
    with right:
        transmission = st.selectbox("Transmission", choices["transmission"])
        drivetrain = st.selectbox("Drivetrain", choices["drivetrain"])
        vehicle_class = st.selectbox("Vehicle class", choices["vehicle_class"])
    engine_description = ""
    model_name = st.text_input("Model designation", max_chars=1000,
                              help="Needed for reviewed Audi and Volvo powertrain checks. Used for prediction when the saved model includes model-name text.")
    if "engine_description" in schema["text_fields"]:
        engine_description = st.text_input("Engine description (optional)", max_chars=1000,
                                           help="Use the available description of the engine or leave this field empty.")
    gasoline = st.checkbox("This vehicle uses gasoline and has no hybrid powertrain")
    submitted = st.form_submit_button("Estimate consumption", type="primary")

if submitted:
    specifications = {"model_year": year, "manufacturer": manufacturer,
                      "displacement_l": displacement, "cylinders": cylinders,
                      "transmission": transmission, "drivetrain": drivetrain,
                      "vehicle_class": vehicle_class, "model_name": model_name,
                      "engine_description": engine_description}
    try:
        result = predict_vehicle(bundle, specifications, gasoline)
    except ValueError as exc:
        st.warning(str(exc))
    else:
        st.metric("Estimated EPA combined consumption", f"{result:.2f} L/100 km")
        st.write("Lower consumption means fewer litres of fuel for the same distance. Actual driving conditions can change consumption.")
