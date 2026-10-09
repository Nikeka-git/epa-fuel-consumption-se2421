"""Validated local inference from project-owned, checksum-verified pipelines."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

from .clean import HYBRID_TEXT
from .powertrain import reviewed_hybrid_rules, reviewed_uncertain_rules, reviewed_missing_engine_rules, requires_model_designation
from .features import CATEGORICAL_FEATURES, MIDTERM_FEATURES, NUMERIC_FEATURES, validate_feature_config
from .split import read_dataset
from .text import ARMS, ARM_TEXT, SANITIZER_VERSION, TEXT_FIELDS, validate_stage_config
from .utils import load_config, project_root, sha256_file


@dataclass
class ModelBundle:
    pipeline: object
    metadata: dict
    config: dict
    interface: dict
    run_dir: Path


def build_interface_schema(training_rows: pd.DataFrame, config: dict, selected_model: str, text_fields=()) -> dict:
    """Category choices and numeric defaults derive exclusively from training rows."""
    if training_rows.empty:
        raise ValueError("Interface requires nonempty training rows")
    if "split" in training_rows and not training_rows["split"].eq("train").all():
        raise ValueError("Interface schema must use training rows only")
    categories = {field: sorted(set(training_rows[field].dropna().astype(str).str.strip()) - {""}) for field in CATEGORICAL_FEATURES}
    categories["vehicle_class"] = [v for v in categories["vehicle_class"] if v in config["scope"]["vehicle_class_allowlist"]]
    defaults = {field: float(pd.to_numeric(training_rows[field], errors="raise").median()) if training_rows[field].notna().any() else None for field in NUMERIC_FEATURES}
    ids = sorted(training_rows["vehicle_id"].astype(str))
    return {"schema_version": "1.0", "selected_model": selected_model, "structured_fields": list(MIDTERM_FEATURES), "text_fields": list(text_fields), "categories": categories, "numeric_defaults": defaults, "allowed_years": list(config["source"]["years"]), "allowed_vehicle_classes": config["scope"]["vehicle_class_allowlist"], "basis": "training rows only", "n_train": len(training_rows), "training_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(), "target_unit": "L/100 km"}


def discover_run(root: str | Path | None = None) -> Path:
    """Prefer the latest complete full Final run, then Midterm, then smoke example."""
    root = Path(root) if root else project_root()
    candidates = []
    for directory in (root / "models", root / "examples" / "smoke" / "models"):
        for metadata_path in directory.glob("*/run_metadata.json"):
            metadata = load_config(metadata_path)
            if metadata.get("status") == "completed" and metadata.get("stage") in {"final", "midterm"}:
                candidates.append((not metadata.get("development_small_dataset", False), metadata["stage"] == "final", metadata.get("completed_at_utc", ""), metadata_path.parent))
    if not candidates:
        raise ValueError("No completed prediction model is available yet")
    return max(candidates, key=lambda row: row[:3])[3]


def _checked_file(path: Path, expected: str | None, label: str) -> None:
    if not expected or not re.fullmatch(r"[a-f0-9]{64}", expected) or sha256_file(path) != expected:
        raise ValueError(f"{label} checksum differs from the saved project artifact")


def load_verified_model(run_dir: str | Path, dataset_path=None, split_dir=None) -> ModelBundle:
    directory = Path(run_dir).resolve()
    metadata = load_config(directory / "run_metadata.json")
    if metadata.get("status") != "completed" or metadata.get("stage") not in {"final", "midterm"}:
        raise ValueError("Only completed project inference runs are supported")
    model = metadata.get("selected_model", "")
    if not re.fullmatch(r"[A-Za-z0-9_]+", model):
        raise ValueError("Invalid selected model name")
    selection_path = directory / "cv_selection.json"
    if metadata.get("cv_selection_sha256"):
        _checked_file(selection_path, metadata["cv_selection_sha256"], "CV selection")
    selection = load_config(selection_path)
    if selection.get("selected_model") != model or selection.get("selection_source") != "train_group_cv" or metadata.get("test_used_for_selection") is not False:
        raise ValueError("Selected model must follow the saved training-CV choice")
    config_path = directory / "config.json"
    _checked_file(config_path, metadata.get("saved_config_sha256"), "Configuration")
    config = load_config(config_path)
    validate_feature_config(config)
    if metadata["stage"] == "final":
        _checked_file(directory / "stage_config.json", metadata.get("saved_stage_config_sha256"), "Stage configuration")
        validate_stage_config(load_config(directory / "stage_config.json"))
        if model not in ARMS or metadata.get("sanitizer_version") != SANITIZER_VERSION:
            raise ValueError("Saved Final text contract differs from this implementation")
    model_path = directory / f"{model}.joblib"
    # Verification precedes deserialization. Only this project's own saved
    # artifacts are supported; hashes are integrity checks, not signatures.
    _checked_file(model_path, metadata.get("model_artifact_sha256", {}).get(model), "Model")
    if metadata.get("interface_schema_sha256"):
        interface_path = directory / "interface_schema.json"
        _checked_file(interface_path, metadata["interface_schema_sha256"], "Interface schema")
        interface = load_config(interface_path)
    else:
        dataset = Path(dataset_path) if dataset_path else Path(metadata["dataset_path"])
        if not dataset.exists() and metadata.get("dataset_relative_path"):
            dataset = project_root() / metadata["dataset_relative_path"].replace("\\", "/")
        split = Path(split_dir) if split_dir else directory.parent.parent / "data" / "splits"
        _checked_file(dataset, metadata.get("dataset_sha256"), "Dataset")
        _checked_file(split / "split_manifest.csv", metadata.get("split_sha256"), "Split")
        manifest = pd.read_csv(split / "split_manifest.csv", dtype={"vehicle_id": str})
        frame = read_dataset(dataset)
        training_ids = set(manifest.loc[manifest["split"].eq("train"), "vehicle_id"])
        if not training_ids or not training_ids.issubset(set(frame["vehicle_id"])):
            raise ValueError("Invalid inference interface training IDs")
        interface = build_interface_schema(frame[frame["vehicle_id"].isin(training_ids)], config, model)
    if interface.get("basis") != "training rows only" or tuple(interface.get("structured_fields", ())) != MIDTERM_FEATURES:
        raise ValueError("Invalid inference interface contract")
    if interface.get("selected_model") != model or any(field not in TEXT_FIELDS for field in interface.get("text_fields", [])):
        raise ValueError("Inference feature set differs from selected model")
    if metadata["stage"] == "final" and tuple(interface["text_fields"]) != ARM_TEXT[model]:
        raise ValueError("Interface text fields differ from the selected Final arm")
    pipeline = joblib.load(model_path)
    if not hasattr(pipeline, "predict"):
        raise ValueError("Saved object does not support prediction")
    return ModelBundle(pipeline, metadata, config, interface, directory)


def validate_specifications(values: dict, config: dict, gasoline_nonhybrid: bool) -> pd.DataFrame:
    if gasoline_nonhybrid is not True:
        raise ValueError("Choose a gasoline car or SUV without a hybrid powertrain")
    allowed = set(MIDTERM_FEATURES + TEXT_FIELDS)
    if set(values) - allowed:
        raise ValueError("Use the requested vehicle specifications only")
    out = {}
    for field in NUMERIC_FEATURES:
        value = values.get(field)
        if value is None and field != "model_year":
            out[field] = np.nan
            continue
        if isinstance(value, bool):
            raise ValueError(f"Enter a valid {field}")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Enter a valid {field}") from exc
        if not np.isfinite(number) or number <= 0 or (field in {"model_year", "cylinders"} and not number.is_integer()):
            raise ValueError(f"Enter a positive valid {field}")
        out[field] = number
    if out["model_year"] not in config["source"]["years"]:
        raise ValueError("Choose a model year from 2015 through 2025")
    if pd.isna(out["displacement_l"]) and pd.isna(out["cylinders"]):
        raise ValueError("Enter engine displacement or cylinder count")
    for field in CATEGORICAL_FEATURES:
        value = values.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 150:
            raise ValueError(f"Enter a valid {field}")
        out[field] = value.strip()
    if out["vehicle_class"] not in config["scope"]["vehicle_class_allowlist"]:
        raise ValueError("Choose a supported car, wagon or SUV class")
    for field in TEXT_FIELDS:
        value = values.get(field, "")
        if value is None:
            value = ""
        if not isinstance(value, str) or len(value) > 1000:
            raise ValueError("Optional vehicle text must be at most 1,000 characters")
        out[field] = value.strip()
    if HYBRID_TEXT.search(" ".join(out[field] for field in TEXT_FIELDS)):
        raise ValueError("Choose a gasoline car or SUV without a hybrid powertrain")
    technology_identity = [out[field] for field in ("manufacturer", "model_name", "model_year", "displacement_l", "cylinders")]
    if requires_model_designation(out["manufacturer"], out["model_year"]) and not out["model_name"]:
        raise ValueError("Enter the model designation so its powertrain can be checked")
    if reviewed_missing_engine_rules(*technology_identity):
        raise ValueError("Enter both engine displacement and cylinder count to resolve this model's powertrain scope")
    if reviewed_hybrid_rules(*technology_identity):
        raise ValueError("This model designation identifies a reviewed hybrid powertrain outside the project's scope")
    if reviewed_uncertain_rules(*technology_identity):
        raise ValueError("This powertrain is outside the validated scope because its source technology labels need clarification")
    return pd.DataFrame([out])


def predict_vehicle(bundle: ModelBundle, specifications: dict, gasoline_nonhybrid: bool = False) -> float:
    frame = validate_specifications(specifications, bundle.config, gasoline_nonhybrid)
    values = np.asarray(bundle.pipeline.predict(frame), dtype=float)
    if values.shape != (1,) or not np.isfinite(values[0]) or values[0] <= 0:
        raise ValueError("This model cannot provide a valid consumption estimate for these specifications")
    return float(values[0])
