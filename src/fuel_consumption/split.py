"""Freeze reproducible vehicle-family train/test and grouped CV assignments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .utils import load_config, project_root, sha256_file, write_json

FROZEN_FILES = ("group_mapping.csv", "group_alias_audit.csv", "split_manifest.csv", "split_coverage.csv", "split_metadata.json")


def read_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={"vehicle_id": str})
    else:
        raise ValueError("Dataset must be .parquet or .csv")
    if "vehicle_id" not in frame:
        raise ValueError("Dataset has no vehicle_id")
    if frame["vehicle_id"].isna().any():
        raise ValueError("Missing vehicle_id is forbidden")
    frame["vehicle_id"] = frame["vehicle_id"].astype(str).str.strip()
    if frame["vehicle_id"].eq("").any() or frame["vehicle_id"].duplicated().any():
        raise ValueError("vehicle_id must be nonempty and unique")
    return frame.sort_values("vehicle_id", kind="stable").reset_index(drop=True)


def normalize_group_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def create_group_mapping(frame: pd.DataFrame, alias_map: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit fallback names without using target or guessing family-name truncations.

    Explicit aliases have manufacturer, model_name, canonical_base_model columns.
    Literal normalized fallback/base-model matches are safe deterministic merges.
    Other potential aliases are reported for review rather than merged heuristically.
    """
    required = {"vehicle_id", "manufacturer", "model_name", "base_model"}
    if not required.issubset(frame):
        raise ValueError(f"Grouping requires {sorted(required - set(frame))}")
    aliases = {}
    if alias_map is not None:
        alias_required = {"manufacturer", "model_name", "canonical_base_model"}
        if not alias_required.issubset(alias_map):
            raise ValueError(f"Alias map requires {sorted(alias_required)}")
        for row in alias_map.to_dict("records"):
            key = (normalize_group_text(row["manufacturer"]), normalize_group_text(row["model_name"]))
            canonical = normalize_group_text(row["canonical_base_model"])
            if not all(key) or not canonical:
                raise ValueError("Empty names in alias map")
            if key in aliases and aliases[key] != canonical:
                raise ValueError(f"Conflicting alias mapping for {key}")
            aliases[key] = canonical
    base_inventory: dict[str, set[str]] = {}
    for row in frame.to_dict("records"):
        make = normalize_group_text(row["manufacturer"])
        base = normalize_group_text(row["base_model"])
        if base:
            base_inventory.setdefault(make, set()).add(base)
    rows, audits = [], []
    for row in frame.to_dict("records"):
        make = normalize_group_text(row["manufacturer"])
        model = normalize_group_text(row["model_name"])
        base = normalize_group_text(row["base_model"])
        if not make or not model:
            raise ValueError("Grouping requires nonempty manufacturer and model_name")
        source = "base_model" if base else "fallback"
        family = base or model
        status = "base_model_present" if base else "unresolved_fallback_review_required"
        if (make, model) in aliases:
            family = aliases[(make, model)]
            source, status = "audited_alias", "explicit_alias_mapping"
        elif not base and model in base_inventory.get(make, set()):
            source, status = "audited_alias", "exact_normalized_match_to_base_model"
        if not base:
            candidates = sorted(candidate for candidate in base_inventory.get(make, set())
                                if model.startswith(candidate + " ") or candidate.startswith(model + " "))
            audits.append({"vehicle_id": row["vehicle_id"], "manufacturer": row["manufacturer"],
                           "model_name": row["model_name"], "normalized_model": model,
                           "canonical_family": family, "status": status,
                           "possible_base_aliases": " | ".join(candidates)})
        # Length prefixes avoid separator collisions between manufacturer/family strings.
        group = f"{len(make)}:{make}|{family}"
        rows.append({"vehicle_id": row["vehicle_id"], "manufacturer": row["manufacturer"],
                     "model_name": row["model_name"], "base_model": row["base_model"],
                     "normalized_manufacturer": make, "canonical_family": family,
                     "model_group": group, "group_source": source})
    audit_columns = ["vehicle_id", "manufacturer", "model_name", "normalized_model", "canonical_family", "status", "possible_base_aliases"]
    return pd.DataFrame(rows), pd.DataFrame(audits, columns=audit_columns)


def validate_manifest(frame: pd.DataFrame, manifest: pd.DataFrame, n_folds: int = 5) -> dict:
    required = {"vehicle_id", "model_group", "split", "cv_fold"}
    if not required.issubset(manifest):
        raise ValueError("Split manifest has missing columns")
    if manifest["vehicle_id"].duplicated().any() or set(manifest["vehicle_id"]) != set(frame["vehicle_id"]):
        raise ValueError("Split manifest IDs must cover dataset exactly once")
    if not set(manifest["split"]).issubset({"train", "test"}):
        raise ValueError("Unknown split label")
    train, test = manifest[manifest["split"].eq("train")], manifest[manifest["split"].eq("test")]
    if train.empty or test.empty:
        raise ValueError("Train and test must be nonempty")
    if set(train["vehicle_id"]) & set(test["vehicle_id"]):
        raise ValueError("Train/test ID overlap")
    if set(train["model_group"]) & set(test["model_group"]):
        raise ValueError("Train/test group overlap")
    if test["cv_fold"].notna().any() or train["cv_fold"].isna().any():
        raise ValueError("Only training rows must have a CV fold")
    expected = set(range(n_folds))
    if set(train["cv_fold"].astype(int)) != expected:
        raise ValueError("Missing or unexpected CV folds")
    fold_sizes = {}
    for fold in range(n_folds):
        valid = train[train["cv_fold"].eq(fold)]
        fitting = train[~train["cv_fold"].eq(fold)]
        if valid.empty or fitting.empty or set(valid["model_group"]) & set(fitting["model_group"]):
            raise ValueError(f"Invalid group overlap or empty CV fold {fold}")
        fold_sizes[str(fold)] = {"n_train": len(fitting), "n_valid": len(valid)}
    return {"n_train": len(train), "n_test": len(test), "n_train_groups": train["model_group"].nunique(),
            "n_test_groups": test["model_group"].nunique(), "test_row_fraction": len(test) / len(manifest),
            "train_test_id_overlap": 0, "train_test_group_overlap": 0, "cv_group_overlap": 0,
            "fold_sizes": fold_sizes}


def _validate_protocol(config: dict) -> None:
    split = config["split"]
    expected = {"method": "GroupShuffleSplit", "test_group_fraction": 0.2, "random_state": 42,
                "cv_method": "GroupKFold", "cv_n_splits": 5, "cv_shuffle": False}
    for key, value in expected.items():
        if split.get(key) != value:
            raise ValueError(f"Split protocol changed: {key} must be {value!r}")
    if split.get("group_fields") != ["manufacturer", "base_model"] or split.get("fallback_group_field") != "model_name":
        raise ValueError("Group definition changed from the frozen project protocol")


def _read_manifest(path: Path) -> pd.DataFrame:
    result = pd.read_csv(path, dtype={"vehicle_id": str, "model_group": str})
    result["cv_fold"] = result["cv_fold"].astype("Int64")
    return result


def load_frozen_split(dataset_path: str | Path, config_path: str | Path, split_dir: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    directory = Path(split_dir) if split_dir else project_root() / "data" / "splits"
    if not all((directory / name).exists() for name in FROZEN_FILES):
        raise ValueError(f"No complete frozen split in {directory}; run fuel_consumption.split first")
    metadata = load_config(directory / "split_metadata.json")
    if metadata["dataset_sha256"] != sha256_file(Path(dataset_path)) or metadata["config_sha256"] != sha256_file(Path(config_path)):
        raise ValueError("Frozen dataset/config fingerprint mismatch; refusing to replace benchmark")
    for name, expected_hash in metadata["artifact_sha256"].items():
        if sha256_file(directory / name) != expected_hash:
            raise ValueError(f"Frozen artifact checksum mismatch: {name}")
    manifest = _read_manifest(directory / "split_manifest.csv")
    frame = read_dataset(dataset_path)
    validate_manifest(frame, manifest, int(metadata["cv_n_splits"]))
    mapping = pd.read_csv(directory / "group_mapping.csv", dtype={"vehicle_id": str})
    joined = manifest.merge(mapping[["vehicle_id", "model_group"]], on="vehicle_id", validate="one_to_one", suffixes=("_manifest", "_mapping"))
    if len(joined) != len(frame) or not joined["model_group_manifest"].eq(joined["model_group_mapping"]).all():
        raise ValueError("Manifest groups differ from frozen mapping")
    return manifest, metadata


def freeze_split(dataset_path: str | Path, config_path: str | Path, output_dir: str | Path | None = None,
                 allow_small: bool = False, alias_map_path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    dataset_path, config_path = Path(dataset_path), Path(config_path)
    config = load_config(config_path)
    _validate_protocol(config)
    frame = read_dataset(dataset_path)
    minimum = config["scope"]["minimum_distinct_rows"]
    if len(frame) < minimum and not allow_small:
        raise ValueError(f"Benchmark requires at least {minimum} rows; --allow-small is only for development smoke runs")
    directory = Path(output_dir) if output_dir else project_root() / "data" / "splits"
    if directory.exists() and any((directory / name).exists() for name in FROZEN_FILES):
        manifest, metadata = load_frozen_split(dataset_path, config_path, directory)
        supplied_alias_hash = sha256_file(Path(alias_map_path)) if alias_map_path else None
        if supplied_alias_hash != metadata.get("alias_map_sha256"):
            raise ValueError("Alias audit changed; refusing to overwrite frozen group mapping")
        return manifest, metadata
    alias_map = pd.read_csv(alias_map_path, keep_default_na=False) if alias_map_path else None
    mapping, audit = create_group_mapping(frame, alias_map)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(splitter.split(frame, groups=mapping["model_group"]))
    if mapping.iloc[train_indices]["model_group"].nunique() < 5:
        raise ValueError("At least five independent training groups are required")
    manifest = mapping[["vehicle_id", "model_group"]].copy()
    manifest["split"] = "test"
    manifest.loc[train_indices, "split"] = "train"
    manifest["cv_fold"] = pd.Series(pd.NA, index=manifest.index, dtype="Int64")
    cv = GroupKFold(n_splits=5)
    train_mapping = mapping.iloc[train_indices]
    for fold, (_, validation) in enumerate(cv.split(train_mapping, groups=train_mapping["model_group"])):
        manifest.loc[train_indices[validation], "cv_fold"] = fold
    checks = validate_manifest(frame, manifest)
    coverage_frame = frame.merge(manifest[["vehicle_id", "split"]], on="vehicle_id", validate="one_to_one")
    coverage = []
    for field in ["model_year", "manufacturer", "vehicle_class"]:
        if field in coverage_frame:
            counts = coverage_frame.groupby(["split", field], dropna=False).size()
            coverage.extend({"split": split, "field": field, "value": str(value), "n": int(n)} for (split, value), n in counts.items())
    group_sizes = manifest.groupby(["split", "model_group"]).size()
    coverage.extend({"split": split, "field": "group_size", "value": group, "n": int(n)} for (split, group), n in group_sizes.items())
    directory.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(directory / "group_mapping.csv", index=False)
    audit.to_csv(directory / "group_alias_audit.csv", index=False)
    manifest.to_csv(directory / "split_manifest.csv", index=False)
    pd.DataFrame(coverage).to_csv(directory / "split_coverage.csv", index=False)
    artifact_hashes = {name: sha256_file(directory / name) for name in FROZEN_FILES if name != "split_metadata.json"}
    metadata = {"schema_version": "1.0", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_path": str(dataset_path.resolve()), "dataset_sha256": sha256_file(dataset_path),
                "config_sha256": sha256_file(config_path), "artifact_sha256": artifact_hashes,
                "alias_map_sha256": sha256_file(Path(alias_map_path)) if alias_map_path else None,
                "group_definition": "normalized manufacturer + base_model; fallback full model_name; across years",
                "method": "GroupShuffleSplit", "test_group_fraction": 0.2, "random_state": 42,
                "cv_method": "GroupKFold", "cv_n_splits": 5, "cv_shuffle": False,
                "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
                "development_small_dataset": len(frame) < minimum,
                "unresolved_fallback_rows": int(audit["status"].eq("unresolved_fallback_review_required").sum()),
                "alias_limitation": "Unresolved full-name fallbacks may represent related families; inspect group_alias_audit.csv before claims of family independence.",
                **checks}
    write_json(directory / "split_metadata.json", metadata)
    return manifest, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/project.json")
    parser.add_argument("--dataset", default="data/processed/vehicles.parquet")
    parser.add_argument("--output", default="data/splits")
    parser.add_argument("--alias-map", help="Reviewed CSV: manufacturer,model_name,canonical_base_model")
    parser.add_argument("--allow-small", action="store_true")
    args = parser.parse_args(argv)
    _, metadata = freeze_split(args.dataset, args.config, args.output, args.allow_small, args.alias_map)
    print(f"Frozen split: {metadata['n_train']} train, {metadata['n_test']} test; groups disjoint; five CV folds")
    if metadata["unresolved_fallback_rows"]:
        print(f"Alias audit: {metadata['unresolved_fallback_rows']} unresolved fallback rows; review group_alias_audit.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
