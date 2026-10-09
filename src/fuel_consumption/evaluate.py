"""Audit saved predictions and regenerate error tables without model fitting."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .split import read_dataset
from .utils import load_config, project_root, sha256_file


def regression_metrics(actual, predicted) -> dict[str, float]:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Predictions and targets must be finite")
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Prediction arrays must be nonempty with equal shapes")
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else np.nan}


def displacement_labels(displacement: pd.Series) -> pd.Series:
    values = pd.to_numeric(displacement, errors="raise")
    labels = pd.cut(values, bins=[0, 2, 3, 4, np.inf], right=True,
                    labels=["(0,2] L", "(2,3] L", "(3,4] L", "(4,+inf) L"])
    return labels.astype(object).where(values.notna(), "Missing").fillna("Invalid")


def error_tables(predictions: pd.DataFrame, dataset: pd.DataFrame, small_threshold: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"vehicle_id", "model", "split", "y_true", "y_pred", "residual", "abs_error"}
    if not required.issubset(predictions):
        raise ValueError(f"Prediction table missing {sorted(required - set(predictions))}")
    if predictions.duplicated(["vehicle_id", "model", "split"]).any():
        raise ValueError("Duplicate prediction rows")
    dataset = dataset.copy()
    dataset["engine_size_bin"] = displacement_labels(dataset["displacement_l"])
    joined = predictions.merge(dataset, on="vehicle_id", how="left", validate="many_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Predictions contain unknown dataset IDs")
    if not np.allclose(joined["y_true"], joined["target_l100km"], rtol=1e-12, atol=1e-12):
        raise ValueError("Saved y_true differs from dataset target")
    expected_residual = joined["y_pred"] - joined["y_true"]
    if not np.allclose(expected_residual, joined["residual"], atol=1e-12, rtol=1e-12):
        raise ValueError("Residual must be prediction minus actual")
    if not np.allclose(expected_residual.abs(), joined["abs_error"], atol=1e-12, rtol=1e-12):
        raise ValueError("Saved absolute errors are inconsistent")
    if not np.isfinite(joined[["y_true", "y_pred", "residual", "abs_error"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite predictions/errors")
    rows = []
    for subgroup in ["vehicle_class", "engine_size_bin"]:
        for (model, split, label), group in joined.groupby(["model", "split", subgroup], dropna=False, observed=True):
            scores = regression_metrics(group["y_true"], group["y_pred"])
            rows.append({"model": model, "split": split, "subgroup_field": subgroup,
                         "subgroup": str(label), "n": len(group), "mae": scores["mae"],
                         "rmse": scores["rmse"], "median_absolute_error": float(group["abs_error"].median()),
                         "bias": float(group["residual"].mean()), "small_support": len(group) < small_threshold})
    columns = ["vehicle_id", "model", "split", "model_year", "manufacturer", "model_name",
               "displacement_l", "cylinders", "transmission", "drivetrain", "vehicle_class",
               "engine_size_bin", "y_true", "y_pred", "residual", "abs_error"]
    difficult = (joined[joined["split"].eq("test")].sort_values(["model", "abs_error", "vehicle_id"],
                 ascending=[True, False, True]).groupby("model", sort=True).head(10))
    return pd.DataFrame(rows), difficult.loc[:, [column for column in columns if column in difficult]]


def _portable_relative_path(value: str) -> Path:
    """Read historical relative metadata paths written on either Windows or POSIX."""
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
        raise ValueError("Artifact metadata must contain a relative path inside its root")
    return Path(*path.parts)


def evaluate_run(run: str | Path, dataset_path: str | Path | None = None, output_root: str | Path | None = None) -> dict:
    run_path = Path(run)
    if not run_path.is_dir():
        artifact_root = Path(output_root) if output_root else project_root()
        run_path = artifact_root / "models" / str(run)
    metadata = load_config(run_path / "run_metadata.json")
    artifact_root = run_path.resolve().parent.parent
    reports = artifact_root / _portable_relative_path(metadata["reports_path"])
    predictions_path = reports / "predictions.csv"
    if sha256_file(predictions_path) != metadata["predictions_sha256"]:
        raise ValueError("Saved predictions checksum changed")
    if dataset_path is None:
        dataset_path = Path(metadata["dataset_path"])
        if not Path(dataset_path).exists() and metadata.get("dataset_relative_path"):
            dataset_path = project_root() / _portable_relative_path(metadata["dataset_relative_path"])
    if sha256_file(Path(dataset_path)) != metadata["dataset_sha256"]:
        raise ValueError("Dataset differs from the training run")
    predictions = pd.read_csv(predictions_path, dtype={"vehicle_id": str})
    dataset = read_dataset(dataset_path)
    config_path = run_path / "config.json"
    if sha256_file(config_path) != metadata["saved_config_sha256"]:
        raise ValueError("Saved training configuration changed")
    config = load_config(config_path)
    subgroup, difficult = error_tables(predictions, dataset, config["evaluation"]["small_subgroup_threshold"])
    subgroup.to_csv(reports / "subgroup_errors.csv", index=False)
    difficult.to_csv(reports / "large_errors.csv", index=False)
    return {"run_id": metadata["run_id"], "reports_path": str(reports),
            "n_prediction_rows": len(predictions), "n_subgroup_rows": len(subgroup),
            "retrained": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run ID or path to existing models/RUN_ID directory")
    parser.add_argument("--dataset", help="Optional relocated original dataset (checksum must match)")
    parser.add_argument("--output-root", help="Artifact root for a run ID; default project root")
    args = parser.parse_args(argv)
    result = evaluate_run(args.run, args.dataset, args.output_root)
    print(f"Rebuilt error tables from {result['n_prediction_rows']} saved predictions in {result['reports_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
