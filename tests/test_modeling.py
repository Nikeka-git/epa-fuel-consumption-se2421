"""Synthetic fixtures test invariants only; they never become project result data."""

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from fuel_consumption.evaluate import displacement_labels, error_tables, evaluate_run
from fuel_consumption.features import MIDTERM_FEATURES, build_pipeline, select_features
from fuel_consumption.split import create_group_mapping, freeze_split, load_frozen_split, read_dataset, validate_manifest
from fuel_consumption.train import run_midterm
from fuel_consumption.utils import project_root


@pytest.fixture
def config():
    return json.loads((project_root() / "configs" / "project.json").read_text(encoding="utf-8"))


@pytest.fixture
def synthetic_frame():
    rows = []
    for group in range(20):
        for variant in range(6):
            displacement = 1.2 + group % 5 * 0.6 + variant * 0.05
            rows.append({"vehicle_id": str(1000 + group * 6 + variant), "model_year": 2015 + variant,
                         "manufacturer": "Maker " + str(group % 3), "model_name": f"Family {group} Trim {variant}",
                         "base_model": f"Family {group}", "displacement_l": displacement,
                         "cylinders": 4 if displacement < 3 else 6,
                         "transmission": "Automatic 6-spd" if variant % 2 else "Manual 6-spd",
                         "drivetrain": "Front-Wheel Drive" if group % 2 else "All-Wheel Drive",
                         "vehicle_class": "Compact Cars" if group % 2 else "Small Sport Utility Vehicle 4WD",
                         "target_l100km": 5 + displacement * 1.4 + variant * 0.01,
                         "combined_mpg": 30.0, "engine_description": "test fixture"})
    return pd.DataFrame(rows)


def _write_inputs(tmp_path, frame, config):
    dataset_path, config_path = tmp_path / "vehicles.csv", tmp_path / "config.json"
    frame.to_csv(dataset_path, index=False)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return dataset_path, config_path


def test_allowlist_blocks_config_leakage_and_drops_raw_columns(config, synthetic_frame):
    assert tuple(select_features(synthetic_frame, config)) == MIDTERM_FEATURES
    for forbidden in ["combined_mpg", "target_l100km", "model_name", "base_model", "vehicle_id"]:
        changed = copy.deepcopy(config)
        changed["features"]["numeric"].append(forbidden)
        with pytest.raises(ValueError, match="allowlist"):
            build_pipeline(changed, "linear")


def test_pipeline_fit_statistics_only_from_training_and_unknown_categories(config, synthetic_frame):
    training = synthetic_frame.iloc[:10].copy()
    training["displacement_l"] = [1.0, 3.0] + [np.nan] * 8
    pipeline = build_pipeline(config, "linear")
    pipeline.fit(training, training["target_l100km"])
    numeric_imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    assert numeric_imputer.statistics_[1] == pytest.approx(2.0)
    held_out = synthetic_frame.iloc[10:12].copy()
    held_out["displacement_l"] = 9999.0
    held_out["manufacturer"] = "Never Seen Maker"
    held_out["vehicle_class"] = "Never Seen Class"
    held_out["transmission"] = pd.NA
    predictions = pipeline.predict(held_out)
    assert np.isfinite(predictions).all()
    assert numeric_imputer.statistics_[1] == pytest.approx(2.0)
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    assert not any(any(forbidden in name for forbidden in ["combined_mpg", "target_l100km", "model_name", "vehicle_id"]) for name in names)
    changed = held_out.copy()
    changed["combined_mpg"] = -98765
    changed["target_l100km"] = 999999
    np.testing.assert_array_equal(predictions, pipeline.predict(changed))


def test_alias_mapping_preserves_full_fallback_and_audits_exact_alias():
    frame = pd.DataFrame([
        {"vehicle_id": "1", "manufacturer": " ACME ", "model_name": " Alpha ", "base_model": "Alpha"},
        {"vehicle_id": "2", "manufacturer": "acme", "model_name": "ALPHA", "base_model": None},
        {"vehicle_id": "3", "manufacturer": "acme", "model_name": "Alpha Sport", "base_model": None},
    ])
    mapping, audit = create_group_mapping(frame)
    assert mapping.iloc[0]["model_group"] == mapping.iloc[1]["model_group"]
    assert mapping.iloc[1]["group_source"] == "audited_alias"
    assert mapping.iloc[2]["canonical_family"] == "alpha sport"
    assert audit.loc[audit["vehicle_id"].eq("3"), "possible_base_aliases"].iloc[0] == "alpha"
    explicit = pd.DataFrame([{"manufacturer": "Acme", "model_name": "Alpha Sport", "canonical_base_model": "Alpha"}])
    audited, _ = create_group_mapping(frame, explicit)
    assert audited.iloc[0]["model_group"] == audited.iloc[2]["model_group"]
    assert audited.iloc[2]["group_source"] == "audited_alias"


def test_group_split_cv_freeze_reproducible_and_config_dataset_refusal(tmp_path, config, synthetic_frame):
    dataset, settings = _write_inputs(tmp_path, synthetic_frame, config)
    directory = tmp_path / "splits"
    with pytest.raises(ValueError, match="at least"):
        freeze_split(dataset, settings, directory)
    first, metadata = freeze_split(dataset, settings, directory, allow_small=True)
    again, metadata_again = freeze_split(dataset, settings, directory, allow_small=True)
    pd.testing.assert_frame_equal(first, again)
    assert metadata == metadata_again
    separate, _ = freeze_split(dataset, settings, tmp_path / "separate", allow_small=True)
    pd.testing.assert_frame_equal(first, separate)
    assert metadata["development_small_dataset"]
    assert metadata["train_test_group_overlap"] == 0
    assert set(first.loc[first["split"].eq("train"), "cv_fold"]) == set(range(5))
    checks = validate_manifest(read_dataset(dataset), first)
    assert checks["cv_group_overlap"] == 0
    changed = synthetic_frame.copy()
    changed.loc[0, "target_l100km"] += 0.1
    changed.to_csv(dataset, index=False)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        freeze_split(dataset, settings, directory, allow_small=True)


def test_frozen_artifacts_tampering_refused(tmp_path, config, synthetic_frame):
    dataset, settings = _write_inputs(tmp_path, synthetic_frame, config)
    directory = tmp_path / "splits"
    freeze_split(dataset, settings, directory, allow_small=True)
    manifest_path = directory / "split_manifest.csv"
    manifest_path.write_text(manifest_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_frozen_split(dataset, settings, directory)


def test_validator_catches_leaked_group_and_missing_fold(tmp_path, config, synthetic_frame):
    dataset, settings = _write_inputs(tmp_path, synthetic_frame, config)
    manifest, _ = freeze_split(dataset, settings, tmp_path / "splits", allow_small=True)
    corrupt = manifest.copy()
    test_index = corrupt.index[corrupt["split"].eq("test")][0]
    corrupt.loc[test_index, "model_group"] = corrupt.loc[corrupt["split"].eq("train"), "model_group"].iloc[0]
    with pytest.raises(ValueError, match="group overlap"):
        validate_manifest(read_dataset(dataset), corrupt)
    corrupt = manifest.copy()
    corrupt.loc[corrupt["split"].eq("train"), "cv_fold"] = pd.NA
    with pytest.raises(ValueError, match="CV fold"):
        validate_manifest(read_dataset(dataset), corrupt)


def test_engine_bins_are_right_closed_and_missing_explicit():
    bins = displacement_labels(pd.Series([1.0, 2.0, 2.01, 3.0, 4.0, 5.0, np.nan]))
    assert bins.tolist() == ["(0,2] L", "(0,2] L", "(2,3] L", "(2,3] L", "(3,4] L", "(4,+inf) L", "Missing"]


def test_train_artifacts_roundtrip_evaluate_no_fit_and_no_overwrite(tmp_path, config, synthetic_frame, monkeypatch):
    dataset, settings = _write_inputs(tmp_path, synthetic_frame, config)
    directory = tmp_path / "splits"
    freeze_split(dataset, settings, directory, allow_small=True)
    result = run_midterm(dataset, settings, "synthetic_test_only", directory, tmp_path / "artifacts", allow_small=True)
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / "models" / "synthetic_test_only"
    reports = artifact_root / result["reports_path"]
    predictions = pd.read_csv(reports / "predictions.csv", dtype={"vehicle_id": str})
    assert len(predictions) == len(synthetic_frame) * 4
    assert predictions.groupby(["model", "vehicle_id"]).size().eq(1).all()
    assert np.allclose(predictions["residual"], predictions["y_pred"] - predictions["y_true"])
    metrics = pd.read_csv(reports / "metrics.csv")
    selection = json.loads((run_dir / "cv_selection.json").read_text())
    assert selection["selected_model"] == metrics.sort_values(["cv_mae_mean", "cv_mae_std"])["model"].iloc[0]
    assert selection["test_evaluation_started"] is False
    assert result["test_used_for_selection"] is False
    assert len(pd.read_csv(reports / "fold_metrics.csv")) == 20
    saved = joblib.load(run_dir / "tree.joblib")
    assert np.isfinite(saved.predict(synthetic_frame.iloc[:2])).all()
    from sklearn.pipeline import Pipeline
    monkeypatch.setattr(Pipeline, "fit", lambda *args, **kwargs: pytest.fail("evaluate must not fit models"))
    # A stored Windows run must remain usable after moving to a POSIX checkout.
    assert "\\" not in result["reports_path"]
    relocated = tmp_path / "portable" / "dataset" / "vehicles.csv"
    relocated.parent.mkdir(parents=True)
    relocated.write_bytes(dataset.read_bytes())
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["reports_path"] = result["reports_path"].replace("/", "\\")
    metadata["dataset_path"] = str(tmp_path / "missing_original.csv")
    metadata["dataset_relative_path"] = "portable\\dataset\\vehicles.csv"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    import fuel_consumption.evaluate as evaluation_module
    monkeypatch.setattr(evaluation_module, "project_root", lambda: tmp_path)
    assert evaluation_module._portable_relative_path("reports\\tables\\run") == Path("reports") / "tables" / "run"
    evaluation = evaluate_run(run_dir)
    assert evaluation["retrained"] is False
    assert pd.read_csv(reports / "subgroup_errors.csv")["small_support"].any()
    corrupt_predictions = predictions.copy()
    corrupt_predictions["residual"] *= -1
    with pytest.raises(ValueError, match="Residual"):
        error_tables(corrupt_predictions, synthetic_frame)
    with pytest.raises(ValueError, match="already exists"):
        run_midterm(dataset, settings, "synthetic_test_only", directory, artifact_root, allow_small=True)
