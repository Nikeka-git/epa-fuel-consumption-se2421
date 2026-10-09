"""Execute the fixed Midterm regression comparison using a frozen grouped split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import re
import time

import joblib
import numpy as np
import pandas as pd

from .evaluate import evaluate_run, regression_metrics
from .features import MIDTERM_FEATURES, MODEL_CLASSES, build_pipeline, dense_memory_estimate, select_features
from .split import load_frozen_split, read_dataset
from .utils import load_config, project_root, sha256_file, write_json


def _prediction_rows(ids: pd.Series, actual, predicted, model: str, split: str, fold=None) -> pd.DataFrame:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    residual = predicted - actual
    return pd.DataFrame({"vehicle_id": ids.to_numpy(), "split": split, "fold": fold,
                         "y_true": actual, "y_pred": predicted, "residual": residual,
                         "abs_error": np.abs(residual), "model": model})


def run_midterm(dataset_path: str | Path, config_path: str | Path, run_id: str,
                split_dir: str | Path | None = None, output_root: str | Path | None = None,
                allow_small: bool = False) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id) or ".." in run_id:
        raise ValueError("Run ID must be a plain name with letters, digits, underscores, dots or hyphens")
    dataset_path, config_path = Path(dataset_path), Path(config_path)
    config = load_config(config_path)
    frame = read_dataset(dataset_path)
    minimum = config["scope"]["minimum_distinct_rows"]
    if len(frame) < minimum and not allow_small:
        raise ValueError(f"At least {minimum} retained rows required; --allow-small labels a development smoke run")
    manifest, split_metadata = load_frozen_split(dataset_path, config_path, split_dir)
    select_features(frame, config)
    y = pd.to_numeric(frame[config["target"]["name"]], errors="raise").astype(float)
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Target must be known, finite, positive; target imputation is forbidden")
    if set(config["midterm_models"]) != set(MODEL_CLASSES):
        raise ValueError("Midterm must include exactly dummy, linear, knn and tree")
    frame = frame.merge(manifest[["vehicle_id", "model_group", "split", "cv_fold"]],
                        on="vehicle_id", how="left", validate="one_to_one", suffixes=("_source", ""))
    train_rows = frame[frame["split"].eq("train")].reset_index(drop=True)
    test_rows = frame[frame["split"].eq("test")].reset_index(drop=True)
    y_train = train_rows[config["target"]["name"]].astype(float)
    X_train = select_features(train_rows, config)
    estimate = dense_memory_estimate(X_train)
    if estimate["upper_bound_float64_bytes"] > 2_000_000_000:
        raise ValueError("Dense feature matrix would exceed 2 GB; review preprocessing before training")
    artifact_root = Path(output_root) if output_root else project_root()
    run_dir = artifact_root / "models" / run_id
    reports = artifact_root / "reports" / "tables" / run_id
    if run_dir.exists() or reports.exists():
        raise ValueError(f"Run {run_id} already exists; choose a new ID rather than overwriting artifacts")
    run_dir.mkdir(parents=True)
    reports.mkdir(parents=True)
    write_json(run_dir / "config.json", config)
    models = list(MODEL_CLASSES)
    fold_metrics, prediction_frames, cv_summary = [], [], []
    started = datetime.now(timezone.utc).isoformat()
    for model in models:
        fold_mae, cv_seconds = [], 0.0
        for fold in range(split_metadata["cv_n_splits"]):
            validation_mask = train_rows["cv_fold"].eq(fold)
            fitting_mask = ~validation_mask
            pipeline = build_pipeline(config, model)
            fitting_groups = set(train_rows.loc[fitting_mask, "model_group"])
            if fitting_groups & set(train_rows.loc[validation_mask, "model_group"]):
                raise ValueError(f"CV group overlap for fold {fold}")
            if model == "knn" and fitting_mask.sum() < config["midterm_models"][model]["params"]["n_neighbors"]:
                raise ValueError("CV training fold smaller than configured KNN n_neighbors; collect more smoke data")
            before = time.perf_counter()
            pipeline.fit(X_train.loc[fitting_mask], y_train.loc[fitting_mask])
            fit_seconds = time.perf_counter() - before
            cv_seconds += fit_seconds
            predictions = pipeline.predict(X_train.loc[validation_mask])
            scores = regression_metrics(y_train.loc[validation_mask], predictions)
            fold_mae.append(scores["mae"])
            fold_metrics.append({"run_id": run_id, "model": model, "fold": fold,
                                 "n_train": int(fitting_mask.sum()), "n_valid": int(validation_mask.sum()),
                                 **scores, "fit_seconds": fit_seconds})
            prediction_frames.append(_prediction_rows(train_rows.loc[validation_mask, "vehicle_id"],
                                                       y_train.loc[validation_mask], predictions, model, "train_oof", fold))
        cv_summary.append({"model": model, "cv_mae_mean": float(np.mean(fold_mae)),
                           "cv_mae_std": float(np.std(fold_mae, ddof=0)), "cv_fit_seconds": cv_seconds})
        print(f"CV {model}: MAE {cv_summary[-1]['cv_mae_mean']:.4f} ± {cv_summary[-1]['cv_mae_std']:.4f} L/100 km")
    selected = min(cv_summary, key=lambda row: (row["cv_mae_mean"], row["cv_mae_std"], models.index(row["model"])))
    # Written before any full-train model predicts test, to make selection auditable.
    selection = {"run_id": run_id, "selected_model": selected["model"],
                 "selected_at_utc": datetime.now(timezone.utc).isoformat(), "selection_source": "train_group_cv",
                 "protocol": "minimum five-fold mean MAE; exact ties use std then fixed model order",
                 "test_evaluation_started": False, "cv_results": cv_summary}
    write_json(run_dir / "cv_selection.json", selection)
    pd.DataFrame(fold_metrics).to_csv(reports / "fold_metrics.csv", index=False)
    X_test = select_features(test_rows, config)
    y_test = test_rows[config["target"]["name"]].astype(float)
    metric_rows = []
    split_path = Path(split_dir) if split_dir else project_root() / "data" / "splits"
    split_hash = sha256_file(split_path / "split_manifest.csv")
    for summary in cv_summary:
        model = summary["model"]
        pipeline = build_pipeline(config, model)
        before = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - before
        predictions = pipeline.predict(X_test)
        scores = regression_metrics(y_test, predictions)
        joblib.dump(pipeline, run_dir / f"{model}.joblib")
        prediction_frames.append(_prediction_rows(test_rows["vehicle_id"], y_test, predictions, model, "test"))
        import json
        metric_rows.append({"stage": "midterm", "run_id": run_id, "model": model,
                            "feature_set": "structured", "dataset_hash": sha256_file(dataset_path),
                            "split_hash": split_hash, "n_train": len(train_rows), "n_test": len(test_rows),
                            "n_folds": split_metadata["cv_n_splits"],
                            "parameters": json.dumps(config["midterm_models"][model]["params"], sort_keys=True),
                            "cv_mae_mean": summary["cv_mae_mean"], "cv_mae_std": summary["cv_mae_std"],
                            "test_mae": scores["mae"], "test_rmse": scores["rmse"], "test_r2": scores["r2"],
                            "fit_seconds": fit_seconds, "cv_fit_seconds": summary["cv_fit_seconds"],
                            "selection_protocol": selection["protocol"], "selected_by_cv": model == selected["model"],
                            "development_small_dataset": len(frame) < minimum})
        print(f"Test {model}: MAE {scores['mae']:.4f} L/100 km")
    metrics = pd.DataFrame(metric_rows)
    dummy = metrics.loc[metrics["model"].eq("dummy")].iloc[0]
    for metric, output in [("cv_mae_mean", "cv_improvement_over_dummy_pct"), ("test_mae", "test_improvement_over_dummy_pct")]:
        metrics[output] = (dummy[metric] - metrics[metric]) / dummy[metric] * 100 if dummy[metric] > 0 else np.nan
    metrics.to_csv(reports / "metrics.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["model", "split", "vehicle_id"])
    predictions.to_csv(reports / "predictions.csv", index=False)
    try:
        relative_dataset = dataset_path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        relative_dataset = None
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "joblib"]}
    metadata = {"schema_version": "1.0", "run_id": run_id, "stage": "midterm", "status": "completed",
                "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_path": str(dataset_path.resolve()), "dataset_relative_path": relative_dataset,
                "dataset_sha256": sha256_file(dataset_path), "config_sha256": sha256_file(config_path),
                "saved_config_sha256": sha256_file(run_dir / "config.json"), "split_sha256": split_hash,
                "split_metadata_sha256": sha256_file(split_path / "split_metadata.json"),
                "group_mapping_sha256": sha256_file(split_path / "group_mapping.csv"),
                "reports_path": (Path("reports") / "tables" / run_id).as_posix(),
                "predictions_sha256": sha256_file(reports / "predictions.csv"),
                "model_artifact_sha256": {model: sha256_file(run_dir / f"{model}.joblib") for model in models},
                "code_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in ["features.py", "split.py", "train.py", "evaluate.py"]},
                "selected_model": selected["model"], "selection_source": "train_group_cv",
                "test_used_for_selection": False, "residual_definition": "prediction_minus_actual",
                "feature_columns": list(MIDTERM_FEATURES), "n_train": len(train_rows), "n_test": len(test_rows),
                "development_small_dataset": len(frame) < minimum,
                "small_dataset_note": "Development smoke results do not fulfill the >=1000-row Midterm requirement" if len(frame) < minimum else None,
                "dense_training_memory_estimate": estimate, "python_version": platform.python_version(),
                "package_versions": versions, "fit_seconds_definition": "fit on full training split; CV time stored separately"}
    write_json(run_dir / "run_metadata.json", metadata)
    evaluate_run(run_dir, dataset_path)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/project.json")
    parser.add_argument("--stage", choices=["midterm"], default="midterm")
    parser.add_argument("--dataset", default="data/processed/vehicles.parquet")
    parser.add_argument("--run", default="midterm_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--output-root", help="Keep development artifacts separate from the benchmark")
    parser.add_argument("--allow-small", action="store_true")
    args = parser.parse_args(argv)
    result = run_midterm(args.dataset, args.config, args.run, args.split_dir, args.output_root, args.allow_small)
    print(f"Completed {result['run_id']}; selected by CV: {result['selected_model']}")
    if result["development_small_dataset"]:
        print(result["small_dataset_note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
