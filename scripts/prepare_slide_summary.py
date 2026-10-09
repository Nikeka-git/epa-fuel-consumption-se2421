"""Prepare slide evidence from one frozen benchmark and saved experiment artifacts.

No network access, new split, model fitting, or invented results. EDA targets use
training rows only. Test subgroup scores remain descriptive diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fuel_consumption.split import load_frozen_split, read_dataset
from fuel_consumption.utils import load_config


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def class_family(value: str) -> str:
    if "Sport Utility" in value:
        return "SUV"
    if "Station Wagon" in value:
        return "Station wagons"
    return "Cars"


def prepare(args) -> dict:
    dataset, split_dir, tables = map(Path, (args.dataset, args.split_dir, args.tables))
    config_path, snapshot_dir = Path(args.config), Path(args.snapshot)
    config = load_config(config_path)
    frame = read_dataset(dataset)
    manifest, split = load_frozen_split(dataset, config_path, split_dir)
    joined = frame.merge(manifest[["vehicle_id", "split"]], on="vehicle_id", validate="one_to_one")
    train = joined[joined["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("No training rows for EDA")
    clean_root = dataset.parent.parent
    cleaned = load_config(clean_root / "interim" / "cleaned_summary.json")
    snapshot = load_config(snapshot_dir / "snapshot.json")
    metrics_path = tables / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    if set(metrics["model"]) != {"dummy", "linear", "knn", "tree"} or len(metrics) != 4:
        raise ValueError("Exactly one saved row for each of the four Midterm models is required")
    if not metrics["dataset_hash"].eq(file_hash(dataset)).all():
        raise ValueError("Metrics dataset hash differs from frozen dataset")
    if not metrics["split_hash"].eq(file_hash(split_dir / "split_manifest.csv")).all():
        raise ValueError("Metrics split hash differs from frozen manifest")
    selected = metrics.loc[metrics["selected_by_cv"].eq(True), "model"].tolist()
    if len(selected) != 1:
        raise ValueError("Saved metrics must identify exactly one model selected by training CV")
    order = {"dummy": 0, "linear": 1, "knn": 2, "tree": 3}
    metrics = metrics.assign(order=metrics["model"].map(order)).sort_values("order").drop(columns="order")
    bins = [0, 2, 3, 4, np.inf]
    labels = ["0–2 L", "2–3 L", "3–4 L", ">4 L"]
    train["engine_bin"] = pd.cut(train["displacement_l"], bins, labels=labels, right=True)
    engine = train.groupby("engine_bin", observed=True)["target_l100km"].agg(n="size", mean_target="mean").reset_index().rename(columns={"engine_bin": "label"})
    train["class_family"] = train["vehicle_class"].map(class_family)
    families = train.groupby("class_family")["target_l100km"].agg(n="size", mean_target="mean").reset_index().rename(columns={"class_family": "label"})
    class_means = train.groupby("vehicle_class")["target_l100km"].agg(n="size", mean_target="mean").reset_index().rename(columns={"vehicle_class": "label"})
    counts, edges = np.histogram(train["target_l100km"].to_numpy(), bins=10)
    hist = [{"label": f"{edges[i]:.1f}–{edges[i+1]:.1f}", "n": int(counts[i]), "lower": float(edges[i]), "upper": float(edges[i+1])} for i in range(len(counts))]
    subgroups_path = tables / "subgroup_errors.csv"
    subgroups = pd.read_csv(subgroups_path)
    selected_subgroups = subgroups[subgroups["model"].eq(selected[0])].copy()
    errors_path = tables / "large_errors.csv"
    errors = pd.read_csv(errors_path, dtype={"vehicle_id": str})
    largest = errors[errors["model"].eq(selected[0]) & errors["split"].eq("test")].sort_values("abs_error", ascending=False).head(3)
    missingness = {key: int(frame[key].isna().sum()) for key in config["features"]["numeric"] + config["features"]["categorical"]}
    limitation = []
    if not snapshot.get("full_catalogue_complete", False):
        limitation.append("The bounded catalogue sample has unequal inclusion probabilities and does not represent vehicle sales.")
    limitation.extend([
        "The target describes EPA estimates. Real driving conditions and driver behaviour can change consumption.",
        "Rounded comb08 values make the converted target discrete. These results assess held-out vehicle families, not future years.",
        "The course reuses this test split across stages. Further model choices use grouped training CV, with test results reported as comparisons.",
    ])
    if cleaned["unresolved_candidate_groups"]:
        limitation.append(f"{cleaned['unresolved_candidate_groups']} duplicate candidate groups still require documented review.")
    if split.get("unresolved_fallback_rows", 0):
        limitation.append(f"{split['unresolved_fallback_rows']} fallback rows still limit claims about independent families.")
    if len(frame) < config["scope"]["minimum_distinct_rows"]:
        limitation.insert(0, "This development sample falls below the project minimum of 1,000 configurations.")
    source_paths = [dataset, metrics_path, subgroups_path, errors_path, split_dir / "split_manifest.csv", split_dir / "split_metadata.json", clean_root / "interim" / "cleaned_summary.json", snapshot_dir / "snapshot.json", config_path]
    return {
        "schema_version": "1.0", "project": config["project"],
        "dataset": {"snapshot_id": snapshot["snapshot_id"], "development_small_dataset": len(frame) < config["scope"]["minimum_distinct_rows"], "years": sorted(int(v) for v in frame["model_year"].dropna().unique()), "requested_years": snapshot["requested_scope"]["years"], "manufacturers_n": int(frame["manufacturer"].nunique()), "manufacturers": sorted(frame["manufacturer"].unique().tolist()), "raw_records": cleaned["raw_vehicle_files"], "cleaned_rows": len(frame), "excluded_records": cleaned["excluded_records"], "duplicates_removed": cleaned["confirmed_duplicate_records_removed"], "unresolved_duplicates": cleaned["unresolved_candidate_groups"], "distinct_count_status": cleaned["distinct_count_status"], "missingness": missingness, "full_catalogue_complete": bool(snapshot.get("full_catalogue_complete", False)), "collection_dates": [snapshot["started_at_utc"], snapshot.get("finished_at_utc")], "collection_mode": snapshot.get("collection_mode"), "request_attempts": snapshot.get("request_attempts_total"), "requested_scope": snapshot["requested_scope"], "source_config": config["source"], "dataset_sha256": file_hash(dataset)},
        "cleaning_stages": records(pd.read_csv(clean_root / "interim" / "cleaning_summary.csv")),
        "split": split,
        "eda": {"scope": "training rows only", "n_train": len(train), "engine_bin_target": records(engine), "class_family_target": records(families), "class_target": records(class_means), "target_hist": hist},
        "metrics": records(metrics), "selected_model": selected[0],
        "subgroups": records(selected_subgroups), "large_errors": records(largest),
        "limitations": limitation,
        "contribution_log": "docs/CONTRIBUTIONS.md",
        "ai_log_text": "OpenAI Codex was used for code implementation, data collection, model experiments, reports and QA. Details are recorded in docs/CONTRIBUTIONS.md.",
        "sources": ["https://www.fueleconomy.gov/feg/ws/index.shtml", "references/Project_Guide.pdf", "references/Midterm_Rubric.docx", "references/Lecture_3.pdf", "references/Lecture_4.pdf", "docs/EXPERIMENT_PROTOCOL.md", "docs/CONTRIBUTIONS.md"],
        "source_files": [{"path": str(path.resolve()), "sha256": file_hash(path)} for path in source_paths],
        "models": config["midterm_models"], "features": config["features"], "evaluation": config["evaluation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = prepare(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Slide summary: {summary['dataset']['cleaned_rows']} rows, {summary['eda']['n_train']} training EDA rows, selected {summary['selected_model']}; {output}")


if __name__ == "__main__":
    main()
