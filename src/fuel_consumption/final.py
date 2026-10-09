"""Compare structured Ridge with preserved text on the same frozen benchmark.

The identical alpha grid uses saved training-group folds for all feature arms.
CV values are selection CV, not an unbiased estimate of the tuning procedure.
Model and alpha selection are durable before any test prediction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time

import joblib
import numpy as np
import pandas as pd

from .evaluate import error_tables, regression_metrics
from .predict import build_interface_schema
from .split import load_frozen_split, read_dataset
from .text import ARMS, ARM_TEXT, SANITIZER_VERSION, build_text_pipeline, validate_stage_config
from .train import _prediction_rows
from .utils import load_config, project_root, sha256_file, write_json


def run_final(dataset_path, config_path, stage_config_path, run_id="final_v1", split_dir=None, output_root=None, allow_small=False) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id) or ".." in run_id:
        raise ValueError("Run ID must be a plain name")
    dataset_path, config_path, stage_config_path = map(Path, (dataset_path, config_path, stage_config_path))
    config, stage = load_config(config_path), load_config(stage_config_path)
    validate_stage_config(stage)
    frame = read_dataset(dataset_path)
    minimum = config["scope"]["minimum_distinct_rows"]
    if len(frame) < minimum and not allow_small:
        raise ValueError(f"At least {minimum} retained rows required; --allow-small labels a development run")
    manifest, split_metadata = load_frozen_split(dataset_path, config_path, split_dir)
    y = pd.to_numeric(frame[config["target"]["name"]], errors="raise").astype(float)
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Known positive finite targets required")
    dataset_for_errors = frame.copy()
    frame = frame.merge(manifest[["vehicle_id", "model_group", "split", "cv_fold"]], on="vehicle_id", validate="one_to_one", suffixes=("_source", ""))
    train = frame[frame["split"].eq("train")].reset_index(drop=True)
    test = frame[frame["split"].eq("test")].reset_index(drop=True)
    artifact_root = Path(output_root) if output_root else project_root()
    run_dir, reports = artifact_root / "models" / run_id, artifact_root / "reports" / "tables" / run_id
    if run_dir.exists() or reports.exists():
        raise ValueError(f"Run {run_id} already exists; choose a new ID")
    run_dir.mkdir(parents=True)
    reports.mkdir(parents=True)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "stage_config.json", stage)
    started = datetime.now(timezone.utc).isoformat()
    grid_rows, all_fold_rows, arm_choices, oof = [], [], [], {}
    for arm in ARMS:
        for alpha in stage["alpha_grid"]:
            fold_rows, frames, elapsed = [], [], 0.0
            for fold in range(split_metadata["cv_n_splits"]):
                valid = train["cv_fold"].eq(fold)
                fit = ~valid
                if set(train.loc[fit, "model_group"]) & set(train.loc[valid, "model_group"]):
                    raise ValueError("Training-CV group overlap")
                pipeline = build_text_pipeline(config, stage, arm, alpha)
                before = time.perf_counter()
                pipeline.fit(train.loc[fit], train.loc[fit, config["target"]["name"]])
                fit_seconds = time.perf_counter() - before
                elapsed += fit_seconds
                predicted = pipeline.predict(train.loc[valid])
                scores = regression_metrics(train.loc[valid, config["target"]["name"]], predicted)
                row = {"run_id": run_id, "model": arm, "alpha": alpha, "fold": fold, "n_train": int(fit.sum()), "n_valid": int(valid.sum()), **scores, "fit_seconds": fit_seconds, "cv_role": stage["cv_role"]}
                fold_rows.append(row)
                all_fold_rows.append(row)
                frames.append(_prediction_rows(train.loc[valid, "vehicle_id"], train.loc[valid, config["target"]["name"]], predicted, arm, "train_oof", fold))
            row = {"model": arm, "alpha": alpha, "cv_mae_mean": float(np.mean([r["mae"] for r in fold_rows])), "cv_mae_std": float(np.std([r["mae"] for r in fold_rows], ddof=0)), "cv_fit_seconds": elapsed, "cv_role": stage["cv_role"]}
            grid_rows.append(row)
            oof[(arm, alpha)] = frames
        choice = min([r for r in grid_rows if r["model"] == arm], key=lambda r: (r["cv_mae_mean"], r["cv_mae_std"], stage["alpha_grid"].index(r["alpha"])))
        arm_choices.append(choice)
        print(f"Selection CV {arm}: alpha={choice['alpha']}, MAE={choice['cv_mae_mean']:.4f} ± {choice['cv_mae_std']:.4f}")
    selected = min(arm_choices, key=lambda r: (r["cv_mae_mean"], r["cv_mae_std"], ARMS.index(r["model"])))
    write_json(run_dir / "cv_selection.json", {"run_id": run_id, "selected_model": selected["model"], "selection_source": "train_group_cv", "selected_at_utc": datetime.now(timezone.utc).isoformat(), "test_evaluation_started": False, "protocol": stage["selection"], "cv_role": stage["cv_role"], "arm_choices": arm_choices, "alpha_grid": stage["alpha_grid"]})
    pd.DataFrame(grid_rows).to_csv(reports / "cv_grid_metrics.csv", index=False)
    pd.DataFrame(all_fold_rows).to_csv(reports / "all_grid_fold_metrics.csv", index=False)
    chosen_folds = [r for r in all_fold_rows if any(r["model"] == c["model"] and r["alpha"] == c["alpha"] for c in arm_choices)]
    pd.DataFrame(chosen_folds).to_csv(reports / "fold_metrics.csv", index=False)
    split_path = Path(split_dir) if split_dir else project_root() / "data" / "splits"
    split_hash = sha256_file(split_path / "split_manifest.csv")
    prediction_frames, metric_rows = [], []
    for choice in arm_choices:
        arm, alpha = choice["model"], choice["alpha"]
        pipeline = build_text_pipeline(config, stage, arm, alpha)
        before = time.perf_counter()
        pipeline.fit(train, train[config["target"]["name"]])
        elapsed = time.perf_counter() - before
        predicted = pipeline.predict(test)
        scores = regression_metrics(test[config["target"]["name"]], predicted)
        joblib.dump(pipeline, run_dir / f"{arm}.joblib")
        prediction_frames.extend(oof[(arm, alpha)])
        prediction_frames.append(_prediction_rows(test["vehicle_id"], test[config["target"]["name"]], predicted, arm, "test"))
        metric_rows.append({"stage": "final", "run_id": run_id, "model": arm, "estimator": "Ridge", "feature_set": arm, "alpha": alpha, "parameters": json.dumps({"alpha": alpha, "solver": stage["solver"]}, sort_keys=True), "dataset_hash": sha256_file(dataset_path), "split_hash": split_hash, "n_train": len(train), "n_test": len(test), "n_folds": split_metadata["cv_n_splits"], "cv_mae_mean": choice["cv_mae_mean"], "cv_mae_std": choice["cv_mae_std"], "cv_role": stage["cv_role"], "test_mae": scores["mae"], "test_rmse": scores["rmse"], "test_r2": scores["r2"], "fit_seconds": elapsed, "cv_fit_seconds": choice["cv_fit_seconds"], "selected_by_cv": arm == selected["model"], "development_small_dataset": len(frame) < minimum})
        print(f"Test {arm}: MAE {scores['mae']:.4f} L/100 km")
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(reports / "metrics.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["model", "split", "vehicle_id"])
    predictions.to_csv(reports / "predictions.csv", index=False)
    subgroup, difficult = error_tables(predictions, dataset_for_errors, config["evaluation"]["small_subgroup_threshold"])
    subgroup.to_csv(reports / "subgroup_errors.csv", index=False)
    difficult.to_csv(reports / "large_errors.csv", index=False)
    test_errors = predictions[predictions["split"].eq("test")].pivot(index="vehicle_id", columns="model", values="abs_error")
    if set(test_errors.index) != set(test["vehicle_id"]) or test_errors.isna().any().any():
        raise ValueError("Paired comparison must cover identical test IDs in all arms")
    paired = test_errors.copy()
    paired["model_group"] = test.set_index("vehicle_id")["model_group"]
    for arm in ARMS[1:]:
        paired[f"{arm}_minus_structured"] = paired[arm] - paired["structured"]
    paired.to_csv(reports / "paired_test_errors.csv")
    paired_summary = [{"text_arm": arm, "baseline": "structured", "n_test": len(test), "mean_abs_error_delta": float(paired[f"{arm}_minus_structured"].mean()), "n_text_better": int((paired[f"{arm}_minus_structured"] < 0).sum()), "interpretation": stage["paired_delta"]} for arm in ARMS[1:]]
    write_json(reports / "paired_comparison.json", {"comparisons": paired_summary, "used_for_selection": False, "inference_limit": "Descriptive paired test differences. No independent-row significance claim or confidence interval."})
    interface = build_interface_schema(train, config, selected["model"], ARM_TEXT[selected["model"]])
    write_json(run_dir / "interface_schema.json", interface)
    try:
        relative_dataset = dataset_path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        relative_dataset = None
    metadata = {"schema_version": "1.0", "run_id": run_id, "stage": "final", "status": "completed", "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_path": str(dataset_path.resolve()), "dataset_relative_path": relative_dataset, "dataset_sha256": sha256_file(dataset_path), "config_sha256": sha256_file(config_path), "saved_config_sha256": sha256_file(run_dir / "config.json"), "stage_config_sha256": sha256_file(stage_config_path), "saved_stage_config_sha256": sha256_file(run_dir / "stage_config.json"), "split_path": str(split_path.resolve()), "split_sha256": split_hash, "split_metadata_sha256": sha256_file(split_path / "split_metadata.json"), "reports_path": (Path("reports") / "tables" / run_id).as_posix(), "predictions_sha256": sha256_file(reports / "predictions.csv"), "model_artifact_sha256": {arm: sha256_file(run_dir / f"{arm}.joblib") for arm in ARMS}, "selected_model": selected["model"], "selected_text_fields": list(ARM_TEXT[selected["model"]]), "selection_source": "train_group_cv", "test_used_for_selection": False, "cv_role": stage["cv_role"], "sanitizer_version": SANITIZER_VERSION, "interface_schema_sha256": sha256_file(run_dir / "interface_schema.json"), "interface_categories_basis": "training rows only", "n_train": len(train), "n_test": len(test), "development_small_dataset": len(frame) < minimum, "residual_definition": "prediction_minus_actual", "python_version": platform.python_version(), "package_versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]}, "code_sha256": {p: sha256_file(Path(__file__).with_name(p)) for p in ["text.py", "final.py", "predict.py", "split.py"]}}
    metadata["cv_selection_sha256"] = sha256_file(run_dir / "cv_selection.json")
    metadata["reports_sha256"] = {name: sha256_file(reports / name) for name in ["metrics.csv", "fold_metrics.csv", "cv_grid_metrics.csv", "all_grid_fold_metrics.csv", "predictions.csv", "subgroup_errors.csv", "large_errors.csv", "paired_test_errors.csv", "paired_comparison.json"]}
    write_json(run_dir / "run_metadata.json", metadata)
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/vehicles.parquet")
    parser.add_argument("--config", default="configs/project.json")
    parser.add_argument("--stage-config", default="configs/final.json")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--run", default="final_v1")
    parser.add_argument("--output-root")
    parser.add_argument("--allow-small", action="store_true")
    args = parser.parse_args(argv)
    result = run_final(args.dataset, args.config, args.stage_config, args.run, args.split_dir, args.output_root, args.allow_small)
    print(f"Completed Final run {result['run_id']}, selected by training CV: {result['selected_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
