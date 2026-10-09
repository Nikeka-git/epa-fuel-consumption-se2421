"""Build the final comparison from saved, hash-checked runs; never fit models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuel_consumption.evaluate import regression_metrics
from fuel_consumption.split import load_frozen_split, read_dataset
from fuel_consumption.utils import load_config, sha256_file, write_json


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


STAGE_MODEL_ORDER = {"midterm": ("dummy", "linear", "knn", "tree"),
                     "endterm": ("random_forest", "extra_trees", "mlp"),
                     "final": ("structured", "structured_model", "structured_engine", "structured_model_engine")}
STAGE_MODELS = {stage: set(models) for stage, models in STAGE_MODEL_ORDER.items()}


def resolve_input(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def close_number(observed, expected, label: str) -> None:
    if not np.isfinite(float(observed)) or not np.isclose(float(observed), float(expected), rtol=1e-10, atol=1e-10):
        raise ValueError(f"Reported value differs from saved predictions: {label}")


def checked_predictions(predictions: pd.DataFrame, metrics: pd.DataFrame, frame: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """Verify un-hashed metric tables by deriving scores from hash-checked rows."""
    required = {"vehicle_id", "model", "split", "fold", "y_true", "y_pred", "residual", "abs_error"}
    if not required.issubset(predictions) or set(predictions.model) != set(metrics.model):
        raise ValueError("Prediction schema/models differ from the metric table")
    if predictions.duplicated(["vehicle_id", "model"]).any() or set(predictions.split) != {"train_oof", "test"}:
        raise ValueError("Prediction IDs must appear exactly once per model and benchmark row")
    joined = predictions.merge(frame[["vehicle_id", "target_l100km"]], on="vehicle_id", validate="many_to_one")
    if len(joined) != len(predictions) or not np.isfinite(joined[["y_true", "y_pred", "residual", "abs_error"]].to_numpy()).all():
        raise ValueError("Unknown IDs or non-finite saved predictions")
    if not np.allclose(joined.y_true, joined.target_l100km, rtol=1e-12, atol=1e-12):
        raise ValueError("Saved prediction targets differ from frozen dataset")
    if not np.allclose(joined.residual, joined.y_pred - joined.y_true, rtol=1e-12, atol=1e-12) or not np.allclose(joined.abs_error, np.abs(joined.residual), rtol=1e-12, atol=1e-12):
        raise ValueError("Saved prediction errors differ from actual/predicted values")
    training = manifest.loc[manifest.split.eq("train")].set_index("vehicle_id")
    test_ids = set(manifest.loc[manifest.split.eq("test"), "vehicle_id"])
    for row in metrics.itertuples():
        saved = predictions.loc[predictions.model.eq(row.model)]
        held_out = saved.loc[saved.split.eq("test")]
        oof = saved.loc[saved.split.eq("train_oof")].set_index("vehicle_id")
        if set(held_out.vehicle_id) != test_ids or set(oof.index) != set(training.index):
            raise ValueError(f"Prediction benchmark coverage differs: {row.model}")
        if held_out.fold.notna().any() or not oof.fold.reindex(training.index).eq(training.cv_fold.astype(float)).all():
            raise ValueError(f"Prediction CV folds differ from frozen manifest: {row.model}")
        scores = regression_metrics(held_out.y_true, held_out.y_pred)
        for name in ("mae", "rmse", "r2"):
            close_number(getattr(row, "test_" + name), scores[name], f"{row.model}/test_{name}")
        fold_mae = [regression_metrics(block.y_true, block.y_pred)["mae"] for _, block in oof.groupby("fold")]
        if len(fold_mae) != 5:
            raise ValueError("Five saved OOF folds are required")
        close_number(row.cv_mae_mean, np.mean(fold_mae), f"{row.model}/cv_mae_mean")
        close_number(row.cv_mae_std, np.std(fold_mae, ddof=0), f"{row.model}/cv_mae_std")


def checked_stage_config_hashes(root: Path, run_dir: Path, metadata: dict) -> tuple[str, str]:
    """Validate a documented context-only edit against original Git bytes.

    Training fingerprints remain in their original fields. Editorial publication
    never permits a change to model grids, preprocessing, or selection settings.
    """
    recorded = (metadata["stage_config_sha256"], metadata["saved_stage_config_sha256"])
    annotation = metadata.get("stage_config_annotation_revision")
    if annotation is None:
        return recorded
    if metadata["stage"] != "endterm" or annotation["evidence_relative_path"] != "evidence/editorial_revision.json":
        raise ValueError("Unsupported stage-configuration annotation revision")
    evidence = load_config(root / annotation["evidence_relative_path"])
    commit = evidence["source_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or commit != annotation["original_commit"]:
        raise ValueError("Invalid editorial source commit")
    records = evidence["stage_config_annotation_revision"]
    expected_names = ["configs/endterm.json", "models/endterm_v1/stage_config.json"]
    if [entry["path"] for entry in records] != expected_names or run_dir.resolve() != (root / "models/endterm_v1").resolve():
        raise ValueError("Unexpected editorial configuration paths")
    published = []
    for entry, recorded_hash in zip(records, recorded):
        original = subprocess.check_output(["git", "show", f"{commit}:{entry['path']}"], cwd=root)
        current_path = root / entry["path"]
        if hashlib.sha256(original).hexdigest() != recorded_hash or entry["recorded_training_sha256"] != recorded_hash:
            raise ValueError("Original training-configuration fingerprint differs")
        old_config, new_config = json.loads(original), load_config(current_path)
        old_context, new_context = old_config.pop("context"), new_config.pop("context")
        if old_config != new_config or not isinstance(new_context, str) or new_context == old_context or entry["changed_json_fields"] != ["context"]:
            raise ValueError("Editorial revision changed computational configuration")
        digest = sha256_file(current_path)
        if entry["published_sha256"] != digest:
            raise ValueError("Published annotation fingerprint differs")
        published.append(digest)
    if published != [annotation["published_stage_config_sha256"], annotation["published_saved_stage_config_sha256"]]:
        raise ValueError("Run annotation fingerprints differ from revision evidence")
    return tuple(published)


def checked_run(root: Path, name: str, dataset_hash: str, split_hash: str, *, allow_development=False,
                config_hash=None, split_metadata_hash=None, frame=None, manifest=None):
    path = Path(name)
    run_dir = path.resolve() if path.is_absolute() else (root / path).resolve() if (root / path).is_dir() else root / "models" / name
    metadata = load_config(run_dir / "run_metadata.json")
    if metadata["status"] != "completed" or (metadata["development_small_dataset"] and not allow_development):
        raise ValueError(f"Not a completed full benchmark: {name}")
    if metadata.get("run_id") != run_dir.name or metadata.get("stage") not in STAGE_MODELS:
        raise ValueError(f"Run identity/stage differs: {name}")
    if metadata["dataset_sha256"] != dataset_hash or metadata["split_sha256"] != split_hash:
        raise ValueError(f"Benchmark differs: {name}")
    expected_source = "train_nested_group_cv" if metadata["stage"] == "endterm" else "train_group_cv"
    if metadata["test_used_for_selection"] is not False or metadata["selection_source"] != expected_source:
        raise ValueError(f"Unexpected selection protocol: {name}")
    if config_hash and metadata["config_sha256"] != config_hash:
        raise ValueError(f"Original project configuration differs: {name}")
    if split_metadata_hash and metadata["split_metadata_sha256"] != split_metadata_hash:
        raise ValueError(f"Original split metadata differs: {name}")
    if set(metadata["model_artifact_sha256"]) != STAGE_MODELS[metadata["stage"]]:
        raise ValueError(f"Predefined stage models/arms are missing: {name}")
    for model, digest in metadata["model_artifact_sha256"].items():
        if not re.fullmatch(r"[A-Za-z0-9_]+", model):
            raise ValueError("Invalid model artifact name")
        if sha256_file(run_dir / f"{model}.joblib") != digest:
            raise ValueError(f"Model checksum differs: {name}/{model}")
    if sha256_file(run_dir / "config.json") != metadata["saved_config_sha256"]:
        raise ValueError(f"Saved configuration differs: {name}")
    if metadata["stage"] != "midterm":
        _, saved_stage_hash = checked_stage_config_hashes(root, run_dir, metadata)
        if sha256_file(run_dir / "stage_config.json") != saved_stage_hash:
            raise ValueError(f"Saved stage configuration differs: {name}")
    selection = load_config(run_dir / "cv_selection.json")
    if selection.get("selected_model") != metadata["selected_model"] or selection.get("selection_source") != expected_source or selection.get("test_evaluation_started") is not False:
        raise ValueError(f"Pre-test selection evidence differs: {name}")
    artifact_root = run_dir.parent.parent
    relative_tables = Path(metadata["reports_path"].replace("\\", "/"))
    if relative_tables.is_absolute() or ".." in relative_tables.parts:
        raise ValueError("Report paths must stay inside their run artifact root")
    tables = (artifact_root / relative_tables).resolve()
    tables.relative_to(artifact_root.resolve())
    if sha256_file(tables / "predictions.csv") != metadata["predictions_sha256"]:
        raise ValueError(f"Predictions checksum differs: {name}")
    if metadata.get("metrics_sha256") and sha256_file(tables / "metrics.csv") != metadata["metrics_sha256"]:
        raise ValueError(f"Metrics checksum differs: {name}")
    metrics = pd.read_csv(tables / "metrics.csv")
    if set(metrics.model) != STAGE_MODELS[metadata["stage"]] or metrics.model.duplicated().any():
        raise ValueError(f"Stage must report exactly its predefined models once: {name}")
    if set(metrics.dataset_hash) != {dataset_hash} or set(metrics.split_hash) != {split_hash}:
        raise ValueError(f"Metric benchmark differs: {name}")
    if not pd.api.types.is_bool_dtype(metrics.selected_by_cv) or metrics.selected_by_cv.sum() != 1 or not metrics.selected_by_cv.eq(metrics.model.eq(metadata["selected_model"])).all():
        raise ValueError(f"Model selection labels differ: {name}")
    order = STAGE_MODEL_ORDER[metadata["stage"]]
    winner = min(metrics.itertuples(), key=lambda row: (row.cv_mae_mean, row.cv_mae_std, order.index(row.model)))
    if winner.model != metadata["selected_model"]:
        raise ValueError("Stage winner differs from its predefined training-CV selection rule")
    if set(metrics.stage) != {metadata["stage"]} or set(metrics.run_id) != {metadata["run_id"]} or not metrics.n_folds.eq(5).all():
        raise ValueError(f"Metric stage/run/folds differ: {name}")
    if metadata["stage"] == "endterm" and not metrics.cv_role.eq("outer_group_cv_of_search_procedure").all():
        raise ValueError("Endterm CV must describe the per-family nested search procedure")
    if metadata["stage"] == "final" and not metrics.cv_role.eq(metadata["cv_role"]).all():
        raise ValueError("Final CV role differs from saved selection-CV metadata")
    metrics["stage"] = metadata["stage"]
    if "cv_role" not in metrics:
        metrics["cv_role"] = "fixed_model_group_cv"
    if frame is not None and manifest is not None:
        predictions = pd.read_csv(tables / "predictions.csv", dtype={"vehicle_id": str})
        checked_predictions(predictions, metrics, frame, manifest)
    return metadata, tables, metrics


def checked_paired(tables: Path, predictions: pd.DataFrame, n_test: int) -> dict:
    paired = load_config(tables / "paired_comparison.json")
    comparisons = paired.get("comparisons", [])
    expected_arms = STAGE_MODELS["final"] - {"structured"}
    if paired.get("used_for_selection") is not False or len(comparisons) != 3 or {row["text_arm"] for row in comparisons} != expected_arms:
        raise ValueError("Paired summary must describe the three predefined text arms without selection")
    errors = predictions.loc[predictions.split.eq("test")].pivot(index="vehicle_id", columns="model", values="abs_error")
    if len(errors) != n_test or errors.isna().any().any():
        raise ValueError("Paired text comparison must cover identical held-out IDs")
    for row in comparisons:
        delta = errors[row["text_arm"]] - errors["structured"]
        if row["baseline"] != "structured" or row["n_test"] != n_test or row["n_text_better"] != int(delta.lt(0).sum()):
            raise ValueError("Paired summary counts/baseline differ from saved predictions")
        close_number(row["mean_abs_error_delta"], delta.mean(), row["text_arm"] + "/paired_delta")
    return paired


def checked_segments(root: Path, metadata: dict, manifest: pd.DataFrame):
    directory = root / "models" / metadata["run_id"] / "segments"
    segment = load_config(directory / "segment_metadata.json")
    if segment != metadata["segments"]:
        raise ValueError("Segment metadata differs from saved Endterm metadata")
    if segment.get("target_used_for_selection") is not False or segment.get("fit_split") != "train_only" or segment.get("selection_source") != "training_silhouette_without_target":
        raise ValueError("Segments must use training X only for fit and selection")
    for filename, digest in segment["artifacts_sha256"].items():
        relative = Path(filename)
        if relative.is_absolute() or len(relative.parts) != 1 or sha256_file(directory / relative) != digest:
            raise ValueError("Segment artifact checksum differs")
    assignments = pd.read_csv(directory / "assignments.csv", dtype={"vehicle_id": str})
    if assignments.vehicle_id.duplicated().any() or set(assignments.vehicle_id) != set(manifest.vehicle_id):
        raise ValueError("Segment assignments must cover benchmark IDs exactly once")
    joined = assignments.merge(manifest[["vehicle_id", "split"]], on="vehicle_id", validate="one_to_one", suffixes=("", "_manifest"))
    if not joined.split.eq(joined.split_manifest).all():
        raise ValueError("Segment train/test assignments differ from frozen manifest")
    return directory, segment, assignments


def build(args):
    root = Path(args.root).resolve()
    output = resolve_input(root, args.output)
    development = getattr(args, "allow_development", False)
    if development and "work" not in output.parts:
        raise ValueError("Development reports must be placed under a work/ directory, never the main report destination")
    if output.exists():
        raise ValueError(f"Report output already exists: {output}")
    dataset = resolve_input(root, getattr(args, "dataset", "data/processed/vehicles.parquet"))
    config = resolve_input(root, getattr(args, "config", "configs/project.json"))
    split_dir = resolve_input(root, getattr(args, "split_dir", "data/splits"))
    frame = read_dataset(dataset)
    manifest, split = load_frozen_split(dataset, config, split_dir)
    if len(frame) < max(1000, load_config(config)["scope"]["minimum_distinct_rows"]) and not development:
        raise ValueError("A full project report requires at least 1,000 retained configurations")
    dataset_hash = sha256_file(dataset)
    split_hash = sha256_file(split_dir / "split_manifest.csv")
    runs = [checked_run(root, name, dataset_hash, split_hash, allow_development=development,
                        config_hash=sha256_file(config), split_metadata_hash=sha256_file(split_dir / "split_metadata.json"),
                        frame=frame, manifest=manifest) for name in args.runs]
    if len(runs) != 3 or {metadata["stage"] for metadata, _, _ in runs} != {"midterm", "endterm", "final"}:
        raise ValueError("Exactly the three project stages must be available")
    cleaning = load_config(resolve_input(root, getattr(args, "cleaning_summary", "data/interim/cleaned_summary.json")))
    if cleaning["processed_parquet_sha256"] != dataset_hash or len(frame) != cleaning["cleaned_rows"]:
        raise ValueError("Cleaning evidence differs from the benchmark")
    if cleaning["unique_vehicle_ids"] != len(frame):
        raise ValueError("Cleaning evidence must report unique retained IDs")
    metrics = pd.concat([table for _, _, table in runs], ignore_index=True)
    if set(metrics.n_train) != {split["n_train"]} or set(metrics.n_test) != {split["n_test"]}:
        raise ValueError("Reported sample sizes differ")
    final_meta, final_tables, _ = next(item for item in runs if item[0]["stage"] == "final")
    paired = checked_paired(final_tables, pd.read_csv(final_tables / "predictions.csv", dtype={"vehicle_id": str}), split["n_test"])
    endterm_meta, endterm_tables, _ = next(item for item in runs if item[0]["stage"] == "endterm")
    endterm_artifact_root = endterm_tables.parent.parent.parent
    segment_dir, segment, assignments = checked_segments(endterm_artifact_root, endterm_meta, manifest)
    groups = pd.read_csv(final_tables / "subgroup_errors.csv")
    groups = groups.loc[groups.model.eq(final_meta["selected_model"]) & groups.split.eq("test")]
    if set(groups.subgroup_field) != {"vehicle_class", "engine_size_bin"}:
        raise ValueError("Missing selected application-model subgroup diagnostics")
    if not groups.groupby("subgroup_field").n.sum().eq(split["n_test"]).all():
        raise ValueError("Application-model subgroup counts do not cover the test rows")
    output.mkdir(parents=True)
    metrics.to_csv(output / "all_models.csv", index=False)
    selected = metrics.loc[metrics.selected_by_cv]
    report = ["# EPA fuel-consumption regression: project results", "",
              "A reproducible machine-learning study of EPA combined fuel consumption. "
              "SE-2421 team: Tsybus Nikita and Bakytzhan Kassymgali.", "",
              "## Data and evaluation", "",
              f"Own individual-record FuelEconomy.gov API collection retained **{len(frame):,} valid configurations after complete-source duplicate checks** "
              f"from {cleaning['raw_vehicle_files']:,} raw vehicle records. "
              f"Years {int(frame.model_year.min())}–{int(frame.model_year.max())}; "
              f"{frame.manufacturer.nunique()} manufacturers and {frame.vehicle_class.nunique()} vehicle classes. "
              "Scope is gasoline, non-hybrid passenger cars, station wagons and SUVs.", "",
              "The target is `235.2145833333333 / comb08`, in L/100 km. "
              "The bounded, seeded catalogue sample has unequal inclusion probabilities and is not sales weighted. "
              "It is not a complete census or a forecast of real road consumption.", "",
              f"One frozen split contains {split['n_train']:,} training rows in {split['n_train_groups']} families "
              f"and {split['n_test']:,} test rows in {split['n_test_groups']} held-out families. "
              "GroupShuffleSplit uses seed 42 and 20% of families; five GroupKFold training folds are reused. "
              "Preprocessing is fitted within training folds; all stages have identical dataset and split hashes.", "",
              "## All saved models", "",
              markdown_table(metrics[["stage", "model", "cv_role", "cv_mae_mean", "cv_mae_std", "test_mae", "test_rmse", "test_r2", "selected_by_cv"]]), "",
              "MAE and RMSE use L/100 km. CV spread is the population standard deviation of five fold MAEs, "
              "not a confidence interval. `selected_by_cv` marks the winner within its stage, chosen before test prediction. "
              "Midterm CV evaluates fixed models. Endterm outer CV evaluates each family’s inner grouped search; "
              "choosing the family from those scores still introduces selection optimism. "
              "Final CV selects Ridge alpha and text arm, so its selection CV is optimistic and is not an unbiased tuning estimate.", "",
              "![Held-out comparison](test_mae_comparison.png)", "", "## Stage decisions", ""]
    if development:
        report[0] = "# Development compatibility check: EPA project report"
        report[2:2] = ["**DEVELOPMENT ONLY: these results do not fulfil the 1,000-row benchmark requirement and are not the main project report.**", ""]
    if cleaning.get("unresolved_candidate_groups", 0):
        report.insert(4, f"**Duplicate review limitation:** {cleaning['unresolved_candidate_groups']} technical-identity candidate groups retain raw differences. "
                      "The distinct count remains provisional unless the independent audit documents their configurations; target conflicts are not silently averaged.")
    for row in selected.itertuples():
        report.append(f"- **{row.stage}: {row.model}**, CV MAE {row.cv_mae_mean:.4f}; held-out MAE {row.test_mae:.4f}, "
                      f"RMSE {row.test_rmse:.4f}, R² {row.test_r2:.4f}.")
    mid_winner = selected.loc[selected.stage.eq("midterm")].iloc[0]
    end_winner = selected.loc[selected.stage.eq("endterm")].iloc[0]
    final_winner = selected.loc[selected.stage.eq("final")].iloc[0]
    dummy = metrics.loc[metrics.stage.eq("midterm") & metrics.model.eq("dummy")].iloc[0]
    ridge_reference = metrics.loc[metrics.stage.eq("final") & metrics.model.eq("structured")].iloc[0]
    report += ["", "## What the comparisons establish", "",
               f"The Midterm CV-selected model reduces held-out MAE by {100 * (1 - mid_winner.test_mae / dummy.test_mae):.1f}% "
               f"relative to the median baseline, with an average absolute error of {mid_winner.test_mae:.4f} L/100 km. "
               "This is useful predictive information for the declared catalogue sample, not a guarantee for an individual car or road trip.", "",
               f"The Endterm CV-selected model changes test MAE by {end_winner.test_mae - mid_winner.test_mae:+.4f} L/100 km "
               "relative to the Midterm CV-selected model. Positive change means worse held-out error. "
               f"The Final CV-selected arm changes test MAE by {final_winner.test_mae - ridge_reference.test_mae:+.4f} L/100 km "
               "relative to structured Ridge. These are descriptive test comparisons; the selections remain those made from training CV. "
               "A lower selection or outer CV score does not guarantee improvement on the fixed test set.", ""]
    report += ["", "## Contribution of text", "",
               "Four equally tuned Ridge arms isolate model names and available engine-description text. "
               "TF-IDF vocabularies fit only each training fold. The deterministic sanitizer removes explicit "
               "fuel-economy, cost, emissions and efficiency markers and their values; original text remains in the source data.", "",
               markdown_table(pd.DataFrame(paired["comparisons"])[["text_arm", "n_test", "mean_abs_error_delta", "n_text_better"]]), "",
               "Delta is text-arm absolute error minus structured-Ridge absolute error on the same test rows: "
               "negative means improvement. This is a descriptive paired comparison, without independent-row significance claims. "
               "It compares Ridge feature sets; it does not establish that text will improve every estimator.", ""]
    # Segment artifacts are written beside Endterm models, not in the tables directory.
    train_pc = assignments.loc[assignments.split.eq("train")]
    figure, ax = plt.subplots(figsize=(9, 6))
    for label, subset in train_pc.groupby("cluster"):
        ax.scatter(subset.pc1, subset.pc2, s=12, alpha=.5, label=f"Cluster {label} (n={len(subset)})")
    ax.set(xlabel="Training PCA component 1", ylabel="Training PCA component 2", title="Vehicle segments fitted on training specifications")
    ax.legend(fontsize=9)
    figure.tight_layout(); figure.savefig(output / "training_segments.png", dpi=150); plt.close(figure)
    report += ["## Vehicle segments and neural network", "",
               f"Training-only PCA retained {segment['pca_components']} components. KMeans selected "
               f"k={segment['selected_k']} by training silhouette {segment['training_silhouette']:.4f} "
               "from the predeclared range 2–6. Scaling and one-hot encoding affect these distances. "
               "Clusters describe configurations, not causal groups or market share; target summaries are calculated after selection.", "",
               "![Training segments](training_segments.png)", "",
               markdown_table(pd.read_csv(segment_dir / "cluster_summary.csv")), "",
               "The MLP uses training-fold target scaling, grouped parameter search and no random-row early-stopping validation split. "
               f"The final MLP fit used {endterm_meta['training_details']['mlp']['n_iter']} iterations; "
               f"the run recorded {endterm_meta['warning_count']} training warnings, including {endterm_meta['convergence_warning_count']} convergence warnings. "
               "All warning records are retained in the Endterm training-warning table.", "", "## Errors, application and limits", "",
               "Per-row predictions, train OOF and test subgroup MAE/bias/sample counts and difficult examples are available "
               "in `reports/tables/<run>/`. Sparse groups are descriptive and should not be used for strong reliability claims. "
               "Associations between specifications and consumption are not causal effects.", "",
               f"The local Streamlit application loads **{final_meta['selected_model']}**, selected within Final by training CV. "
               "It validates specifications, uses training-only category options, checks saved-model hashes and returns L/100 km. "
               "It is the controlled Final text model, not a claim of being the globally best architecture. "
               "Missing optional text remains empty; predictions depend on the catalogue’s coverage.", "",
               "Blank EPA technology labels do not independently certify a non-hybrid powertrain. "
               "Manufacturer-reviewed corrections and explicit uncertainty quarantine were fixed before splitting; "
               "further source omissions may remain. Seven auxiliary source encodings are unconfirmed and excluded from model inputs. "
               "The reviewed family aliases preserve conservative source taxonomy, not certified physical generations or platforms. "
               "See the [data card](../../docs/DATA_CARD.md) and [powertrain review](../../docs/POWERTRAIN_REVIEW.md).", "",
               "All stages reuse the same test set as requested. Test results are comparisons, not a fresh independent confirmation "
               "of a later development process. No parameter, seed or feature choice is made from these reported test scores. "
               "Rounded source MPG creates a discrete converted target. The three stages implement the project’s "
               "predeclared progression from structured regressors to tuned models and controlled text comparisons.", ""]
    report += ["## Descriptive errors of the application model", "",
               "The following groups come from saved test diagnostics of the Final CV-selected arm. "
               "`small_support=True` means fewer than 30 observations; such groups do not establish a stable reliability ranking.", "",
               markdown_table(groups.loc[groups.subgroup_field.eq("vehicle_class")].sort_values("mae", ascending=False).head(5)
                              [["subgroup", "n", "mae", "bias", "small_support"]]), "",
               markdown_table(groups.loc[groups.subgroup_field.eq("engine_size_bin")]
                              [["subgroup", "n", "mae", "bias", "small_support"]]), ""]
    figure, ax = plt.subplots(figsize=(10, 7))
    labels = metrics.stage + ": " + metrics.model
    ax.barh(labels, metrics.test_mae, color=["#2874a6" if v else "#9cb9cc" for v in metrics.selected_by_cv])
    ax.invert_yaxis(); ax.set(xlabel="Held-out MAE (L/100 km)", title="Same held-out vehicle families across all stages")
    figure.tight_layout(); figure.savefig(output / "test_mae_comparison.png", dpi=150); plt.close(figure)
    (output / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    write_json(output / "experiment_summary.json", {
        "built_at_utc": datetime.now(timezone.utc).isoformat(), "cleaning": cleaning, "split": split,
        "runs": [metadata for metadata, _, _ in runs], "selected_models": json.loads(selected.to_json(orient="records")),
        "paired_text_comparison": paired, "segments": segment,
        "dataset_sha256": dataset_hash, "split_sha256": split_hash,
        "development_report": development, "model_count": len(metrics), "stage_count": len(runs),
        "report_builder_sha256": sha256_file(Path(__file__)),
        "input_metrics_sha256": {metadata["run_id"]: sha256_file(tables / "metrics.csv") for metadata, tables, _ in runs},
        "test_used_for_selection": False, "artifacts_sha256": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({"output": str(output), "rows": len(frame), "models": len(metrics)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--runs", nargs=3, default=["midterm_v1", "endterm_v1", "final_v1"])
    parser.add_argument("--output", default="reports/project_v1")
    parser.add_argument("--dataset", default="data/processed/vehicles.parquet")
    parser.add_argument("--config", default="configs/project.json")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--cleaning-summary", default="data/interim/cleaned_summary.json")
    parser.add_argument("--allow-development", action="store_true", help="Allow labelled private smoke checks; --output must be under work/")
    build(parser.parse_args())
