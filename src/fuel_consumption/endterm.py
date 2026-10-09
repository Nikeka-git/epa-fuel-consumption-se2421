"""Nested grouped ensemble/MLP comparison using the unchanged frozen benchmark.

Per-family outer OOF evaluates the train-only parameter search. The full-training
winning search score is reported separately as selection CV, never as unbiased
CV. Selecting a family by these outer scores adds selection optimism; it is not
an unbiased estimate of a complete family-selection procedure. Families and
hyperparameters are fixed before any test predictions.
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
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .evaluate import _portable_relative_path, error_tables, regression_metrics
from .features import MIDTERM_FEATURES, StrictFeatureSelector, build_preprocessor, dense_memory_estimate, select_features
from .segments import fit_segments
from .split import load_frozen_split, read_dataset
from .train import _prediction_rows
from .utils import load_config, project_root, sha256_file, write_json


MODEL_CLASSES = {"random_forest": RandomForestRegressor, "extra_trees": ExtraTreesRegressor, "mlp": MLPRegressor}
SEARCH_COLUMNS = ["model", "context", "candidate", "fold", "n_train", "n_valid", "train_groups", "valid_groups",
                  "group_overlap", "mae", "rmse", "r2", "fit_seconds", "parameters"]
WARNING_COLUMNS = ["model", "context", "candidate", "fold", "category", "message"]


def candidate_parameters(stage_config: dict, model: str) -> list[dict]:
    settings = stage_config["models"][model]
    if settings["class"] != MODEL_CLASSES[model].__name__:
        raise ValueError(f"Unexpected Endterm estimator class: {model}")
    candidates = settings.get("candidates")
    maximum = int(stage_config["evaluation"]["max_candidates_per_model"])
    if not 1 <= maximum <= 12 or not isinstance(candidates, list) or not 1 <= len(candidates) <= maximum:
        raise ValueError("Each Endterm model requires 1..12 bounded preselected candidates")
    parameters = [{**settings.get("params", {}), **candidate} for candidate in candidates]
    if len({json.dumps(item, sort_keys=True) for item in parameters}) != len(parameters):
        raise ValueError("Duplicate Endterm candidates waste the fixed search budget")
    for item in parameters:
        if item.get("random_state") is None:
            raise ValueError("Endterm estimators require an explicit random_state")
        if model == "mlp":
            if item.get("early_stopping", False) is not False:
                raise ValueError("MLP random row validation is forbidden; early_stopping must be false")
            if not 1 <= int(item.get("max_iter", 200)) <= 1000:
                raise ValueError("MLP max_iter must be bounded to 1..1000")
            if settings.get("target_transform") != "train_fold_standard_scaler":
                raise ValueError("MLP target scaling must fit inside every training fold")
        elif item.get("n_jobs", 1) != 1 or not 1 <= int(item.get("n_estimators", 100)) <= 500:
            raise ValueError("Ensembles require n_jobs=1 and 1..500 trees per bounded fit")
        MODEL_CLASSES[model](**item).get_params()  # Reject unknown constructor parameters before the run starts.
    return parameters


def build_endterm_pipeline(config: dict, stage_config: dict, model: str, parameters: dict) -> Pipeline:
    select_config_parameters = candidate_parameters(stage_config, model)
    if parameters not in select_config_parameters:
        raise ValueError("Estimator parameters must belong to the preselected candidate list")
    estimator = MODEL_CLASSES[model](**parameters)
    if model == "mlp":
        estimator = TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())
    return Pipeline([( "select", StrictFeatureSelector()), ("preprocess", build_preprocessor(config)), ("model", estimator)])


def _fit_with_warnings(pipeline, X, y, model, context, candidate, fold, warning_rows) -> float:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        pipeline.fit(X, y)
    warning_rows.extend({"model": model, "context": context, "candidate": candidate, "fold": fold,
                         "category": item.category.__name__, "message": str(item.message)} for item in caught)
    return time.perf_counter() - started


def _search(X: pd.DataFrame, y: pd.Series, groups: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]],
            config: dict, stage: dict, model: str, context: str, search_rows: list, warning_rows: list) -> dict:
    """A small explicit grouped search with fitted preprocessing inside each split."""
    candidates = []
    for candidate, parameters in enumerate(candidate_parameters(stage, model)):
        scores = []
        for fold, (fitting, validation) in enumerate(folds):
            fitting_groups, validation_groups = set(groups.iloc[fitting]), set(groups.iloc[validation])
            if fitting_groups & validation_groups:
                raise ValueError("Endterm search has train/validation group overlap")
            pipeline = build_endterm_pipeline(config, stage, model, parameters)
            fit_seconds = _fit_with_warnings(pipeline, X.iloc[fitting], y.iloc[fitting], model, context,
                                             candidate, fold, warning_rows)
            with threadpool_limits(limits=1):
                predicted = pipeline.predict(X.iloc[validation])
            metrics = regression_metrics(y.iloc[validation], predicted)
            scores.append(metrics["mae"])
            search_rows.append({"model": model, "context": context, "candidate": candidate, "fold": fold,
                                "n_train": len(fitting), "n_valid": len(validation),
                                "train_groups": len(fitting_groups), "valid_groups": len(validation_groups),
                                "group_overlap": 0, **metrics, "fit_seconds": fit_seconds,
                                "parameters": json.dumps(parameters, sort_keys=True)})
        candidates.append({"candidate": candidate, "parameters": parameters,
                           "mae_mean": float(np.mean(scores)), "mae_std": float(np.std(scores, ddof=0)),
                           "n_folds": len(folds)})
    selected = min(candidates, key=lambda row: (row["mae_mean"], row["mae_std"], row["candidate"]))
    return {"context": context, "model": model, "selected": selected, "candidates": candidates}


def _midterm_comparison(midterm_run: Path | None, dataset_hash: str, split_hash: str) -> tuple[pd.DataFrame | None, dict | None]:
    if midterm_run is None:
        return None, None
    metadata = load_config(midterm_run / "run_metadata.json")
    if metadata.get("stage") != "midterm" or metadata.get("status") != "completed" or metadata.get("test_used_for_selection") is not False:
        raise ValueError("Comparison requires an original completed Midterm run selected without test")
    if metadata["dataset_sha256"] != dataset_hash or metadata["split_sha256"] != split_hash:
        raise ValueError("Midterm comparison must use exactly the same dataset and frozen split")
    table_path = midterm_run.resolve().parent.parent / _portable_relative_path(metadata["reports_path"])
    if sha256_file(table_path / "predictions.csv") != metadata["predictions_sha256"]:
        raise ValueError("Midterm predictions checksum changed")
    metrics = pd.read_csv(table_path / "metrics.csv")
    if not metrics.dataset_hash.eq(dataset_hash).all() or not metrics.split_hash.eq(split_hash).all():
        raise ValueError("Midterm metrics fingerprints differ")
    if metrics.model.eq(metadata["selected_model"]).sum() != 1:
        raise ValueError("Cannot identify the original train-CV-selected Midterm model")
    return metrics, metadata


def run_endterm(dataset_path: str | Path, config_path: str | Path, stage_config_path: str | Path, run_id: str,
                split_dir: str | Path | None = None, output_root: str | Path | None = None,
                midterm_run: str | Path | None = None, allow_small: bool = False) -> dict:
    """Run nested grouped search, fixed-test evaluation and train-only segments."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id) or ".." in run_id:
        raise ValueError("Run ID must be a plain name without parent-directory components")
    dataset_path, config_path, stage_config_path = Path(dataset_path), Path(config_path), Path(stage_config_path)
    split_dir = Path(split_dir) if split_dir else project_root() / "data" / "splits"
    config, stage = load_config(config_path), load_config(stage_config_path)
    if stage.get("stage") != "endterm" or set(stage.get("models", {})) != set(MODEL_CLASSES):
        raise ValueError("Endterm requires random_forest, extra_trees and mlp in its separate stage config")
    if stage["evaluation"].get("method") != "nested_group_cv" or stage["evaluation"].get("test_used_for_selection") is not False:
        raise ValueError("Endterm requires nested grouped CV and no test selection")
    for model in MODEL_CLASSES:
        candidate_parameters(stage, model)
    frame = read_dataset(dataset_path)
    minimum = max(1000, int(config["scope"]["minimum_distinct_rows"]))
    if len(frame) < minimum and not allow_small:
        raise ValueError(f"At least {minimum} retained rows required; --allow-small is a labelled development check")
    manifest, split = load_frozen_split(dataset_path, config_path, split_dir)
    if split["cv_n_splits"] != 5:
        raise ValueError("Endterm reuses the five frozen training folds")
    frame = frame.merge(manifest[["vehicle_id", "model_group", "split", "cv_fold"]], on="vehicle_id", validate="one_to_one", suffixes=("_source", ""))
    train, test = [frame.loc[frame["split"].eq(label)].reset_index(drop=True) for label in ["train", "test"]]
    target = config["target"]["name"]
    if not np.isfinite(frame[target].astype(float)).all() or frame[target].astype(float).le(0).any():
        raise ValueError("Endterm target must remain known, finite and positive")
    X, y, groups = select_features(train, config), train[target].astype(float), train.model_group
    estimate = dense_memory_estimate(X)
    if estimate["upper_bound_float64_bytes"] > 2_000_000_000:
        raise ValueError("Dense feature upper bound exceeds 2 GB")
    outer_folds = [(np.flatnonzero(train.cv_fold.ne(fold)), np.flatnonzero(train.cv_fold.eq(fold))) for fold in range(5)]
    inner_n = int(stage["evaluation"]["inner_n_splits"])
    if not 2 <= inner_n <= 5 or any(groups.iloc[fit].nunique() < inner_n for fit, _ in outer_folds):
        raise ValueError("Every outer training portion needs enough distinct groups for 2..5 inner GroupKFold splits")
    dataset_hash, split_hash = sha256_file(dataset_path), sha256_file(split_dir / "split_manifest.csv")
    config_hash, stage_hash = sha256_file(config_path), sha256_file(stage_config_path)
    old_metrics, old_metadata = _midterm_comparison(Path(midterm_run) if midterm_run else None, dataset_hash, split_hash)
    artifact_root = Path(output_root) if output_root else project_root()
    run_dir, reports = artifact_root / "models" / run_id, artifact_root / "reports" / "tables" / run_id
    if run_dir.exists() or reports.exists():
        raise ValueError(f"Run {run_id} already exists; never overwrite an experiment")
    run_dir.mkdir(parents=True)
    reports.mkdir(parents=True)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "stage_config.json", stage)
    started = datetime.now(timezone.utc).isoformat()
    search_rows, warning_rows, outer_rows, predictions, searches, summaries = [], [], [], [], [], []
    model_order = list(MODEL_CLASSES)
    try:
        for model in model_order:
            fold_scores = []
            for fold, (fitting, validation) in enumerate(outer_folds):
                if set(groups.iloc[fitting]) & set(groups.iloc[validation]):
                    raise ValueError("Frozen outer CV groups overlap")
                outer_X, outer_y, outer_groups = X.iloc[fitting].reset_index(drop=True), y.iloc[fitting].reset_index(drop=True), groups.iloc[fitting].reset_index(drop=True)
                inner_folds = list(GroupKFold(n_splits=inner_n).split(outer_X, groups=outer_groups))
                search = _search(outer_X, outer_y, outer_groups, inner_folds, config, stage, model,
                                 f"outer_{fold}", search_rows, warning_rows)
                searches.append(search)
                best = search["selected"]
                pipeline = build_endterm_pipeline(config, stage, model, best["parameters"])
                fit_seconds = _fit_with_warnings(pipeline, X.iloc[fitting], y.iloc[fitting], model,
                                                 f"outer_refit_{fold}", best["candidate"], fold, warning_rows)
                with threadpool_limits(limits=1):
                    predicted = pipeline.predict(X.iloc[validation])
                scores = regression_metrics(y.iloc[validation], predicted)
                fold_scores.append(scores["mae"])
                outer_rows.append({"run_id": run_id, "model": model, "fold": fold, "n_train": len(fitting),
                                   "n_valid": len(validation), "group_overlap": 0, **scores, "fit_seconds": fit_seconds,
                                   "selected_parameters": json.dumps(best["parameters"], sort_keys=True),
                                   "cv_role": "outer_group_cv_of_search_procedure"})
                predictions.append(_prediction_rows(train.iloc[validation].vehicle_id, y.iloc[validation], predicted, model, "train_oof", fold))
            full_search = _search(X, y, groups, outer_folds, config, stage, model, "full_train_selection", search_rows, warning_rows)
            searches.append(full_search)
            summaries.append({"model": model, "cv_mae_mean": float(np.mean(fold_scores)),
                              "cv_mae_std": float(np.std(fold_scores, ddof=0)),
                              "selection_cv_mae_mean": full_search["selected"]["mae_mean"],
                              "selection_cv_mae_std": full_search["selected"]["mae_std"],
                              "parameters": full_search["selected"]["parameters"],
                              "candidate": full_search["selected"]["candidate"]})
            pd.DataFrame(search_rows, columns=SEARCH_COLUMNS).to_csv(reports / "search_fold_metrics.csv", index=False)
            pd.DataFrame(outer_rows).to_csv(reports / "fold_metrics.csv", index=False)
            pd.DataFrame(warning_rows, columns=WARNING_COLUMNS).to_csv(reports / "training_warnings.csv", index=False)
            write_json(run_dir / "parameter_searches.json", searches)
            print(f"Nested CV {model}: MAE {summaries[-1]['cv_mae_mean']:.4f} ± {summaries[-1]['cv_mae_std']:.4f} L/100 km")
        selection_key = lambda row: (row["cv_mae_mean"], row["cv_mae_std"], model_order.index(row["model"]))
        selected = min(summaries, key=selection_key)
        selected_ensemble = min([row for row in summaries if row["model"] != "mlp"], key=selection_key)
        selection = {"selected_model": selected["model"], "selected_ensemble_model": selected_ensemble["model"],
                     "selection_source": "train_nested_group_cv", "selected_at_utc": datetime.now(timezone.utc).isoformat(),
                     "test_evaluation_started": False, "test_used_for_selection": False,
                     "protocol": stage["evaluation"], "cv_results": summaries}
        write_json(run_dir / "cv_selection.json", selection)  # Durable before touching any test predictions.
        X_test, y_test = select_features(test, config), test[target].astype(float)
        metric_rows, training_details = [], {}
        for summary in summaries:
            model = summary["model"]
            pipeline = build_endterm_pipeline(config, stage, model, summary["parameters"])
            fit_seconds = _fit_with_warnings(pipeline, X, y, model, "full_train_refit", summary["candidate"], None, warning_rows)
            with threadpool_limits(limits=1):
                predicted = pipeline.predict(X_test)
            scores = regression_metrics(y_test, predicted)
            joblib.dump(pipeline, run_dir / f"{model}.joblib", compress=3)
            predictions.append(_prediction_rows(test.vehicle_id, y_test, predicted, model, "test"))
            metric_rows.append({"stage": "endterm", "run_id": run_id, "model": model, "feature_set": "structured",
                                "dataset_hash": dataset_hash, "split_hash": split_hash, "n_train": len(train), "n_test": len(test),
                                "n_folds": 5, "cv_role": "outer_group_cv_of_search_procedure",
                                "cv_mae_mean": summary["cv_mae_mean"], "cv_mae_std": summary["cv_mae_std"],
                                "selection_cv_mae_mean": summary["selection_cv_mae_mean"], "selection_cv_mae_std": summary["selection_cv_mae_std"],
                                "test_mae": scores["mae"], "test_rmse": scores["rmse"], "test_r2": scores["r2"],
                                "parameters": json.dumps(summary["parameters"], sort_keys=True), "fit_seconds": fit_seconds,
                                "selected_by_cv": model == selected["model"], "selected_ensemble_by_cv": model == selected_ensemble["model"],
                                "development_small_dataset": len(frame) < minimum})
            if model == "mlp":
                fitted = pipeline.named_steps["model"].regressor_
                training_details[model] = {"n_iter": int(fitted.n_iter_), "final_training_loss_scaled_y": float(fitted.loss_),
                                           "early_stopping": False, "loss_curve": [float(item) for item in getattr(fitted, "loss_curve_", [])]}
            print(f"Test {model}: MAE {scores['mae']:.4f} L/100 km")
        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(reports / "metrics.csv", index=False)
        predicted_frame = pd.concat(predictions, ignore_index=True).sort_values(["model", "split", "vehicle_id"])
        predicted_frame.to_csv(reports / "predictions.csv", index=False)
        errors, difficult = error_tables(predicted_frame, read_dataset(dataset_path), config["evaluation"]["small_subgroup_threshold"])
        errors.to_csv(reports / "subgroup_errors.csv", index=False)
        difficult.to_csv(reports / "large_errors.csv", index=False)
        pd.DataFrame(warning_rows, columns=WARNING_COLUMNS).to_csv(reports / "training_warnings.csv", index=False)
        comparison = None
        if old_metrics is not None:
            old = old_metrics.copy()
            old["cv_role"] = "fixed_model_group_cv"
            columns = ["stage", "run_id", "model", "cv_role", "cv_mae_mean", "cv_mae_std", "test_mae", "test_rmse", "test_r2", "selected_by_cv"]
            pd.concat([old[columns], metrics[columns]], ignore_index=True).to_csv(reports / "midterm_comparison.csv", index=False)
            old_selected = old.loc[old.model.eq(old_metadata["selected_model"])].iloc[0]
            new_selected = metrics.loc[metrics.model.eq(selected["model"])].iloc[0]
            comparison = {"midterm_run_id": old_metadata["run_id"], "midterm_selected_model": old_metadata["selected_model"],
                          "midterm_test_mae": float(old_selected.test_mae), "endterm_selected_model": selected["model"],
                          "endterm_test_mae": float(new_selected.test_mae),
                          "test_mae_improvement_l100km": float(old_selected.test_mae - new_selected.test_mae),
                          "comparison_is_descriptive_only": True, "test_used_for_selection": False}
        with threadpool_limits(limits=1):
            segments = fit_segments(train, test, config, stage["segments"], run_dir / "segments")
        if (sha256_file(dataset_path), sha256_file(config_path), sha256_file(stage_config_path),
                sha256_file(split_dir / "split_manifest.csv")) != (dataset_hash, config_hash, stage_hash, split_hash):
            raise ValueError("Frozen dataset, split or configuration changed during the Endterm run")
        versions = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "joblib", "threadpoolctl"]}
        result = {"schema_version": "1.0", "stage": "endterm", "run_id": run_id, "status": "completed",
                  "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "dataset_path": str(dataset_path.resolve()), "dataset_sha256": dataset_hash,
                  "config_sha256": config_hash, "saved_config_sha256": sha256_file(run_dir / "config.json"),
                  "stage_config_sha256": stage_hash, "saved_stage_config_sha256": sha256_file(run_dir / "stage_config.json"),
                  "split_sha256": split_hash, "split_metadata_sha256": sha256_file(split_dir / "split_metadata.json"),
                  "reports_path": (Path("reports") / "tables" / run_id).as_posix(),
                  "predictions_sha256": sha256_file(reports / "predictions.csv"),
                  "metrics_sha256": sha256_file(reports / "metrics.csv"),
                  "model_artifact_sha256": {name: sha256_file(run_dir / f"{name}.joblib") for name in model_order},
                  "code_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in ["endterm.py", "segments.py", "features.py", "split.py", "evaluate.py"]},
                  "selected_model": selected["model"], "selected_ensemble_model": selected_ensemble["model"],
                  "selection_source": "train_nested_group_cv", "test_used_for_selection": False,
                  "cv_note": "Per-family outer frozen train folds estimate its inner GroupKFold tuning procedure. Choosing the family by these outer scores adds selection optimism; the winning score is not unbiased for full family selection. Full-train winning search is selection CV only.",
                  "inner_n_splits": inner_n, "outer_n_splits": 5, "feature_columns": list(MIDTERM_FEATURES),
                  "n_train": len(train), "n_test": len(test), "residual_definition": "prediction_minus_actual",
                  "warning_count": len(warning_rows), "convergence_warning_count": sum(row["category"] == "ConvergenceWarning" for row in warning_rows),
                  "training_details": training_details, "segments": segments, "midterm_comparison": comparison,
                  "development_small_dataset": len(frame) < minimum, "dense_training_memory_estimate": estimate,
                  "python_version": platform.python_version(), "package_versions": versions}
        write_json(run_dir / "run_metadata.json", result)
        return result
    except BaseException as error:
        write_json(run_dir / "run_metadata.json", {"stage": "endterm", "run_id": run_id, "status": "interrupted",
                   "started_at_utc": started, "interrupted_at_utc": datetime.now(timezone.utc).isoformat(),
                   "dataset_sha256": dataset_hash, "split_sha256": split_hash, "error_type": type(error).__name__, "error": str(error)})
        pd.DataFrame(warning_rows, columns=WARNING_COLUMNS).to_csv(reports / "training_warnings.csv", index=False)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/vehicles.parquet")
    parser.add_argument("--config", default="configs/project.json")
    parser.add_argument("--stage-config", default="configs/endterm.json")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--run", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--midterm-run", help="Original completed models/RUN_ID on exactly the same dataset/split")
    parser.add_argument("--allow-small", action="store_true", help="Label a technical check; never counts as sufficient benchmark data")
    args = parser.parse_args(argv)
    result = run_endterm(args.dataset, args.config, args.stage_config, args.run, args.split_dir,
                         args.output_root, args.midterm_run, args.allow_small)
    print(f"Completed {result['run_id']}; selected by nested train CV: {result['selected_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
