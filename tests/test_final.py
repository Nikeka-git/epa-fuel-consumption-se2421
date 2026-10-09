"""Synthetic invariants for Final text comparison and local inference."""
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fuel_consumption.final import run_final
from fuel_consumption.predict import ModelBundle, build_interface_schema, discover_run, load_verified_model, predict_vehicle, validate_specifications
from fuel_consumption.split import freeze_split
from fuel_consumption.text import ARMS, SafeTextVectorizer, build_text_pipeline, sanitize_text
from fuel_consumption.utils import project_root


@pytest.fixture
def config():
    return json.loads((project_root() / "configs/project.json").read_text())


@pytest.fixture
def stage():
    result = json.loads((project_root() / "configs/final.json").read_text())
    result["alpha_grid"] = [0.1, 1.0]
    return result


@pytest.fixture
def frame():
    rows = []
    for family in range(20):
        for variant in range(6):
            displacement = 1.3 + family % 5 * 0.6 + variant * 0.04
            rows.append({"vehicle_id": str(1000 + family * 6 + variant), "model_year": 2015 + variant,
                         "manufacturer": f"Maker {family % 3}", "base_model": f"Family {family}",
                         "model_name": f"Family {family} EcoBoost Trim {variant}",
                         "engine_description": "GUZZLER turbo" if variant % 2 else None,
                         "displacement_l": displacement, "cylinders": 4 if displacement < 3 else 6,
                         "transmission": "Automatic (S6)", "drivetrain": "Front-Wheel Drive",
                         "vehicle_class": "Compact Cars", "target_l100km": 5 + displacement * 1.4 + variant * 0.01})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("leak", ["GUZZLER", "Gas Guzzler", "27 MPG", "MPG: 27", "29 MPGe", "9.2 L/100 km", "L/100km=9.2", "fuel economy: 27 MPG", "EPA combined: 29", "annual fuel cost: $2,500.00", "fuelCost08=2500", "CO2: 190 g/km", "198 g/mi", "emissions score: 8", "ghgScore=7", "feScore=9", "efficiency score: 7", "smartwayScore=9", "youSaveSpend=-1500"])
def test_sanitizer_removes_explicit_target_proxies_and_preserves_engine_terms(leak):
    clean = sanitize_text(f"EcoBoost V8 S4 {leak} Turbo AWD")
    assert "EcoBoost V8 S4" in clean
    assert "Turbo AWD" in clean
    assert sanitize_text(leak).strip(" :=-/") == ""


def test_sanitizer_missing_idempotent_and_unsupported_type():
    for empty in [None, pd.NA, np.nan, ""]:
        assert sanitize_text(empty) == ""
    value = "EcoBoost   GUZZLER 27 MPG Turbo"
    assert sanitize_text(sanitize_text(value)) == sanitize_text(value)
    with pytest.raises(ValueError, match="strings"):
        sanitize_text(123)


def test_sanitizer_removes_suffixed_measurement_keys_and_values():
    assert sanitize_text("DOHC fuelCostA=2,100 co2TailpipeGpmA=250 barrels08=10.5") == "DOHC"


def test_text_vocabulary_uses_training_only_and_handles_empty_fold():
    vectorizer = SafeTextVectorizer().fit(pd.Series(["Alpha Turbo", "Beta EcoBoost GUZZLER"]))
    before = dict(vectorizer.vectorizer_.vocabulary_)
    vectorizer.transform(pd.Series(["validationonly 88 MPG"]));
    assert vectorizer.vectorizer_.vocabulary_ == before
    assert "validationonly" not in before and "guzzler" not in before
    empty = SafeTextVectorizer().fit(pd.Series([None, "GUZZLER 27 MPG"]))
    assert empty.empty_training_vocabulary_
    assert empty.transform(pd.Series(["new unseen engine", None])).shape == (2, 1)
    assert empty.transform(pd.Series(["new unseen engine"])).nnz == 0


@pytest.mark.parametrize("arm", ARMS)
def test_final_pipeline_drops_unapproved_inputs_and_preserves_originals(config, stage, frame, arm):
    original = frame.copy(deep=True)
    pipeline = build_text_pipeline(config, stage, arm, 1.0).fit(frame.iloc[:80], frame.iloc[:80]["target_l100km"])
    predictions = pipeline.predict(frame.iloc[80:])
    changed = frame.iloc[80:].copy()
    changed["combined_mpg"] = 999999.0
    changed["target_l100km"] = -1e12
    changed["vehicle_id"] = "999999"
    np.testing.assert_array_equal(predictions, pipeline.predict(changed))
    pd.testing.assert_frame_equal(frame, original)
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    assert not any("target_l100km" in x or "combined_mpg" in x or "vehicle_id" in x for x in names)


def test_interface_rejects_test_rows_and_preserves_training_category_choices(config, frame):
    train = frame.iloc[:12].copy()
    train["split"] = "train"
    interface = build_interface_schema(train, config, "linear")
    assert interface["n_train"] == 12 and interface["basis"] == "training rows only"
    train.loc[train.index[0], "split"] = "test"
    with pytest.raises(ValueError, match="training rows only"):
        build_interface_schema(train, config, "linear")


def spec():
    return {"model_year": 2020, "manufacturer": "Maker 1", "displacement_l": 2.0, "cylinders": 4,
            "transmission": "Automatic (S6)", "drivetrain": "Front-Wheel Drive", "vehicle_class": "Compact Cars", "model_name": "EcoBoost", "engine_description": "Turbo"}


@pytest.mark.parametrize("field,value", [("model_year", 2026), ("model_year", True), ("model_year", 2020.5), ("displacement_l", 0), ("displacement_l", np.inf), ("cylinders", 4.5), ("vehicle_class", "Standard Pickup Trucks 4WD"), ("model_name", "Prius Hybrid"), ("engine_description", "x" * 1001), ("manufacturer", "")])
def test_prediction_input_scope_validation(config, field, value):
    values = spec()
    values[field] = value
    with pytest.raises(ValueError):
        validate_specifications(values, config, True)


def test_prediction_requires_gasoline_confirmation_and_does_not_clip_invalid_output(config):
    with pytest.raises(ValueError, match="gasoline"):
        validate_specifications(spec(), config, False)
    values = spec()
    values["combined_mpg"] = 99
    with pytest.raises(ValueError, match="requested"):
        validate_specifications(values, config, True)
    class BadPredictor:
        def predict(self, X):
            return np.asarray([-1.0])
    bundle = ModelBundle(BadPredictor(), {}, config, {}, Path("."))
    with pytest.raises(ValueError, match="valid consumption"):
        predict_vehicle(bundle, spec(), True)
    values = spec()
    values["displacement_l"] = values["cylinders"] = None
    with pytest.raises(ValueError, match="displacement or cylinder"):
        validate_specifications(values, config, True)


def test_final_saved_comparison_same_ids_selection_before_test_and_verified_load(tmp_path, config, stage, frame, monkeypatch):
    dataset, settings, stage_path = tmp_path / "vehicles.csv", tmp_path / "project.json", tmp_path / "final.json"
    frame.to_csv(dataset, index=False)
    settings.write_text(json.dumps(config)); stage_path.write_text(json.dumps(stage))
    split_dir = tmp_path / "splits"
    manifest, _ = freeze_split(dataset, settings, split_dir, allow_small=True)
    output = tmp_path / "artifacts"
    run = "synthetic_final_only"
    original_predict = Pipeline.predict
    def audited_predict(self, X, *a, **kw):
        if X["split"].eq("test").all():
            selection = json.loads((output / "models" / run / "cv_selection.json").read_text())
            assert selection["test_evaluation_started"] is False
        return original_predict(self, X, *a, **kw)
    monkeypatch.setattr(Pipeline, "predict", audited_predict)
    result = run_final(dataset, settings, stage_path, run, split_dir, output, allow_small=True)
    monkeypatch.setattr(Pipeline, "predict", original_predict)
    tables = output / result["reports_path"]
    predictions = pd.read_csv(tables / "predictions.csv", dtype={"vehicle_id": str})
    assert len(predictions) == len(frame) * 4
    for arm in ARMS:
        assert set(predictions.loc[predictions["model"].eq(arm) & predictions["split"].eq("test"), "vehicle_id"]) == set(manifest.loc[manifest["split"].eq("test"), "vehicle_id"])
    paired = pd.read_csv(tables / "paired_test_errors.csv")
    assert len(paired) == result["n_test"]
    for arm in ARMS[1:]:
        np.testing.assert_allclose(paired[f"{arm}_minus_structured"], paired[arm] - paired["structured"])
    assert len(pd.read_csv(tables / "all_grid_fold_metrics.csv")) == 4 * 2 * 5
    run_dir = output / "models" / run
    bundle = load_verified_model(run_dir)
    assert bundle.interface["n_train"] == result["n_train"]
    assert np.isfinite(predict_vehicle(bundle, spec(), True))
    assert discover_run(output) == run_dir
    with pytest.raises(ValueError, match="already exists"):
        run_final(dataset, settings, stage_path, run, split_dir, output, allow_small=True)
    selected_model_path = run_dir / f"{result['selected_model']}.joblib"
    selected_model_path.write_bytes(selected_model_path.read_bytes() + b"tampered")
    monkeypatch.setattr(joblib, "load", lambda *a, **kw: pytest.fail("Cannot deserialize before verifying the model checksum"))
    with pytest.raises(ValueError, match="Model checksum"):
        load_verified_model(run_dir)
