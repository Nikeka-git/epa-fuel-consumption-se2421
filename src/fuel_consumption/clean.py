"""Conservative EPA vehicle normalization and an auditable dataset builder.

Raw responses are evidence, never the training feature matrix. This module retains
text and provenance for later stages; train.py must select its explicit allowlist.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .utils import load_config, project_root, sha256_file, write_json


FIELD_MAP = {
    "vehicle_id": "id", "model_year": "year", "manufacturer": "make",
    "model_name": "model", "base_model": "baseModel", "displacement_l": "displ",
    "cylinders": "cylinders", "transmission": "trany", "drivetrain": "drive",
    "vehicle_class": "VClass", "engine_description": "eng_dscr", "engine_id": "engId",
    "atv_type": "atvType", "fuel_type_primary": "fuelType1",
    "fuel_type_secondary": "fuelType2", "electric_motor": "evMotor",
    "phev_blended": "phevBlended", "combined_mpg": "comb08",
    "combined_mpg_unrounded": "comb08U",
}
NORMALIZED_COLUMNS = list(FIELD_MAP) + [
    "target_l100km", "source_url", "fetched_at_utc", "raw_sha256",
    "snapshot_id", "model_group", "audit_flags",
]
HYBRID_TEXT = re.compile(r"\b(?:hev|phev|mhev|hybrid|eassist)\b", re.IGNORECASE)
CANDIDATE_FIELDS = [
    "model_year", "manufacturer", "model_name", "displacement_l", "cylinders",
    "transmission", "drivetrain", "vehicle_class", "engine_description",
    "engine_id", "fuel_type_primary",
]
FILTER_ORDER = [
    "raw_integrity", "required_identity", "model_year", "powertrain_metadata",
    "primary_fuel", "secondary_fuel", "alternative_technology", "electric_motor",
    "phev", "hybrid_text", "vehicle_class", "target", "engine_specs",
]


def _string(value: Any, field: str, reasons: list[str]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        reasons.append(f"invalid_type:{field}")
        return None
    return value.strip() or None


def _number(value: Any, field: str, reasons: list[str], *, integer: bool = False) -> float | int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        reasons.append(f"invalid_numeric:{field}")
        return None
    try:
        parsed = float(value)
    except (ValueError, TypeError, OverflowError):
        reasons.append(f"invalid_numeric:{field}")
        return None
    if not math.isfinite(parsed) or (integer and not parsed.is_integer()):
        reasons.append(f"invalid_numeric:{field}")
        return None
    return int(parsed) if integer else parsed


def _alias(record: dict, main: str, alias: str, reasons: list[str], flags: list[str]) -> Any:
    if main in record and alias in record and record[main] != record[alias]:
        reasons.append(f"conflicting_alias:{main}")
    if main in record:
        return record[main]
    if alias in record:
        flags.append(f"compatibility_alias:{alias}->{main}")
        return record[alias]
    return None


def _group_text(value: str) -> str:
    return " ".join(value.casefold().split())


def normalize_record(record: dict, provenance: dict, config: dict) -> tuple[dict, list[str], list[str]]:
    """Return normalized data, every exclusion reason, and nonfatal audit flags.

    Empty source strings become nulls; missing powertrain *keys* are quarantined.
    provenance supplies URL, received_at_utc, SHA-256, and snapshot_id. File/hash
    validation happens in build_dataset, where exact source bytes are available.
    """
    reasons: list[str] = []
    flags: list[str] = []
    out = dict.fromkeys(NORMALIZED_COLUMNS)
    if not isinstance(record, dict):
        return out, ["invalid_record_object"], flags
    scope = config["scope"]
    atv = _alias(record, "atvType", "atvtype", reasons, flags)
    base = _alias(record, "baseModel", "basemodel", reasons, flags)
    for key, source in FIELD_MAP.items():
        if key in {"model_year", "displacement_l", "cylinders", "combined_mpg", "combined_mpg_unrounded", "phev_blended", "vehicle_id"}:
            continue
        value = atv if key == "atv_type" else base if key == "base_model" else record.get(source)
        out[key] = _string(value, source, reasons)
    source_id = record.get("id")
    if isinstance(source_id, int) and not isinstance(source_id, bool):
        source_id = str(source_id)
    if isinstance(source_id, str) and re.fullmatch(r"[1-9]\d*", source_id.strip()):
        out["vehicle_id"] = source_id.strip()
    else:
        reasons.append("invalid_vehicle_id")
    for key in ("model_year", "displacement_l", "cylinders", "combined_mpg", "combined_mpg_unrounded"):
        out[key] = _number(record.get(FIELD_MAP[key]), FIELD_MAP[key], reasons, integer=key in {"model_year", "cylinders"})
    for required in ("manufacturer", "model_name", "vehicle_class"):
        if not out[required]:
            reasons.append(f"missing_required:{required}")
    if out["model_year"] not in config["source"]["years"]:
        reasons.append("model_year_out_of_scope")
    for field in ("fuelType1", "fuelType2", "evMotor", "phevBlended"):
        if field not in record:
            reasons.append(f"missing_powertrain_metadata:{field}")
        elif record[field] is None:
            reasons.append(f"null_powertrain_metadata:{field}")
    if "atvType" not in record and "atvtype" not in record:
        reasons.append("missing_powertrain_metadata:atvType")
    elif atv is None:
        reasons.append("null_powertrain_metadata:atvType")
    if out["fuel_type_primary"] not in scope["fuel_type_primary_allowlist"]:
        reasons.append("primary_fuel_out_of_scope")
    if out["fuel_type_secondary"]:
        reasons.append("secondary_fuel_present")
    if out["atv_type"]:
        reasons.append("alternative_technology_present")
    if out["electric_motor"]:
        reasons.append("electric_motor_present")
    phev_raw = record.get("phevBlended")
    if phev_raw is None:
        out["phev_blended"] = None
    elif isinstance(phev_raw, str) and not phev_raw.strip():
        out["phev_blended"] = False
    elif isinstance(phev_raw, bool):
        out["phev_blended"] = phev_raw
    elif isinstance(phev_raw, str) and phev_raw.strip().casefold() in {"true", "1", "y", "yes", "false", "0", "n", "no"}:
        out["phev_blended"] = phev_raw.strip().casefold() in {"true", "1", "y", "yes"}
    else:
        reasons.append("unknown_phev_boolean")
    if out["phev_blended"]:
        reasons.append("phev_blended")
    if HYBRID_TEXT.search(" ".join([out["model_name"] or "", out["engine_description"] or ""])):
        reasons.append("hybrid_text_signal")
    if out["vehicle_class"] not in scope["vehicle_class_allowlist"]:
        reasons.append("vehicle_class_out_of_scope")
    mpg = out["combined_mpg"]
    if mpg is None or mpg <= 0:
        reasons.append("invalid_target")
    else:
        out["target_l100km"] = config["target"]["us_mpg_conversion_constant"] / mpg
        if not math.isfinite(out["target_l100km"]) or out["target_l100km"] <= 0:
            out["target_l100km"] = None
            reasons.append("invalid_target")
    for field in ("displacement_l", "cylinders"):
        if out[field] is not None and out[field] <= 0:
            reasons.append(f"nonpositive_engine_spec:{field}")
        elif out[field] is None and not any(reason.endswith(":" + FIELD_MAP[field]) for reason in reasons):
            flags.append(f"missing_optional:{field}")
    if out["displacement_l"] is None and out["cylinders"] is None:
        reasons.append("insufficient_engine_specs")
    for field in ("transmission", "drivetrain", "engine_description", "base_model"):
        if out[field] is None:
            flags.append(f"missing_optional:{field}")
    if out["manufacturer"] and (out["base_model"] or out["model_name"]):
        group_name = out["base_model"] or out["model_name"]
        out["model_group"] = _group_text(out["manufacturer"]) + "::" + _group_text(group_name)
        if not out["base_model"]:
            flags.append("model_group_fallback:model_name")
    out.update({
        "source_url": provenance.get("url") or provenance.get("source_url"),
        "fetched_at_utc": provenance.get("received_at_utc") or provenance.get("fetched_at_utc"),
        "raw_sha256": provenance.get("sha256") or provenance.get("raw_sha256"),
        "snapshot_id": provenance.get("snapshot_id"),
        "audit_flags": "|".join(dict.fromkeys(flags)),
    })
    return out, list(dict.fromkeys(reasons)), list(dict.fromkeys(flags))


def _stage(reason: str) -> str:
    if reason.startswith(("raw_", "missing_provenance", "invalid_record", "filename_")):
        return "raw_integrity"
    if reason == "invalid_vehicle_id" or reason.startswith("missing_required"):
        return "required_identity"
    if "year" in reason:
        return "model_year"
    if "powertrain_metadata" in reason or "conflicting_alias" in reason or reason.startswith("invalid_type"):
        return "powertrain_metadata"
    for prefix, stage in (("primary_fuel", "primary_fuel"), ("secondary_fuel", "secondary_fuel"), ("alternative_technology", "alternative_technology"), ("electric_motor", "electric_motor"), ("phev", "phev"), ("unknown_phev", "phev"), ("hybrid_text", "hybrid_text"), ("vehicle_class", "vehicle_class")):
        if reason.startswith(prefix):
            return stage
    if reason == "invalid_target" or reason.endswith(":comb08"):
        return "target"
    return "engine_specs"


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _complete_comparable_record(record: dict) -> dict:
    """Only known record identifiers/dates are removed; all other raw keys matter.

    Candidate formation never uses target. Complete equality is an additional
    conservative confirmation step, including agreement of all observed outputs.
    """
    comparable = {key: value for key, value in record.items() if key not in {"id", "createdOn", "modifiedOn"}}
    if isinstance(comparable.get("emissionsList"), dict):
        emissions = dict(comparable["emissionsList"])
        details = emissions.get("emissionsInfo")
        if isinstance(details, list):
            emissions["emissionsInfo"] = [{key: value for key, value in row.items() if key != "id"} if isinstance(row, dict) else row for row in details]
        elif isinstance(details, dict):
            emissions["emissionsInfo"] = {key: value for key, value in details.items() if key != "id"}
        comparable["emissionsList"] = emissions
    return comparable


def _disposition(raw_key: str, config: dict) -> str:
    reverse = {value: key for key, value in FIELD_MAP.items()}
    internal = reverse.get(raw_key)
    if internal in config["features"]["numeric"] + config["features"]["categorical"]:
        return "structured_predictor"
    if internal in config["features"]["final_text"]:
        return "preserve_original_text_for_final_sanitization"
    if raw_key in {"comb08", "comb08U"}:
        return "target_or_target_audit_only"
    if re.match(r"(?i)^(?:city|highway|comb|UCity|UHighway|co2|fuelCost|barrels|range|phevCity|phevComb|phevHwy)", raw_key) or raw_key in {"feScore", "ghgScore", "ghgScoreA", "guzzler", "youSaveSpend", "emissionsList"}:
        return "excluded_from_predictors_leakage_or_output"
    return "source_audit_only_excluded_from_predictors"


def build_dataset(snapshot_dir: Path, output_root: Path, config: dict) -> dict:
    """Build CSV/parquet plus full exclusion, duplicate, category, schema audits.

    output_root is the project directory. This function supports small development
    snapshots and reports minimum_rows_met honestly. CLI enforces the minimum.
    It refuses to replace a processed table belonging to another snapshot.
    """
    import pandas as pd

    snapshot_dir, output_root = Path(snapshot_dir), Path(output_root)
    snapshot_id = snapshot_dir.name
    snapshot_metadata = snapshot_dir / "snapshot.json"
    if snapshot_metadata.exists():
        document = json.loads(snapshot_metadata.read_text(encoding="utf-8"))
        declared_id = document.get("snapshot_id") if isinstance(document, dict) else None
        if not isinstance(declared_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", declared_id) or declared_id in {".", ".."}:
            raise ValueError("snapshot.json must contain a valid snapshot_id; refusing to infer a different provenance identity.")
        snapshot_id = declared_id
    interim = output_root / "data" / "interim"
    processed = output_root / "data" / "processed"
    existing_schema = processed / "schema.json"
    if existing_schema.exists():
        prior = json.loads(existing_schema.read_text(encoding="utf-8"))
        if prior.get("snapshot_id") != snapshot_id:
            raise ValueError("Processed dataset belongs to another snapshot; select a separate output root for a new benchmark.")
        if prior.get("source_config") != config:
            raise ValueError("Processed dataset uses another configuration; select a separate output root for a new benchmark.")
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict] = {}
    request_log = snapshot_dir / "requests.jsonl"
    if request_log.exists():
        for line in request_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            filename = entry.get("file")
            if filename:
                metadata[str(filename).replace("\\", "/")] = entry
    cache_file = snapshot_dir / "cache_index.json"
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
        cache_entries = cache.get("entries", cache)
        for entry in cache_entries.values():
            if isinstance(entry, dict) and entry.get("file"):
                metadata.setdefault(str(entry["file"]).replace("\\", "/"), entry)
    collector_inventory = None
    inventory_path = snapshot_dir / "inventory.json"
    if inventory_path.exists():
        inventory_document = json.loads(inventory_path.read_text(encoding="utf-8"))
        collector_inventory = inventory_document.get("records") if isinstance(inventory_document, dict) else None
        if not isinstance(collector_inventory, dict):
            raise ValueError("Collector inventory.json must contain a records mapping; refusing an unauditable snapshot.")
    accepted: list[dict] = []
    exclusions: list[dict] = []
    inventory: list[dict] = []
    raw_by_id: dict[str, dict] = {}
    all_raw_records: list[dict] = []
    all_reasons: list[list[str]] = []
    field_types: dict[str, Counter] = defaultdict(Counter)
    missing_fields: Counter = Counter()
    raw_keys: set[str] = set()
    files = sorted((snapshot_dir / "vehicles").glob("*.json"), key=lambda path: path.name)
    if not files:
        raise ValueError(f"No individual records found in {snapshot_dir / 'vehicles'}")
    for path in files:
        relative = path.relative_to(snapshot_dir).as_posix()
        provenance = dict(metadata.get(relative, {}), snapshot_id=snapshot_id)
        integrity: list[str] = []
        actual_hash = sha256_file(path)
        if not provenance.get("sha256") or actual_hash != provenance.get("sha256"):
            integrity.append("raw_checksum_missing_or_mismatch")
        if provenance.get("status") != 200 or provenance.get("valid") is False:
            integrity.append("raw_request_not_validated")
        if not provenance.get("url") or not (provenance.get("received_at_utc") or provenance.get("fetched_at_utc")):
            integrity.append("missing_provenance")
        try:
            record = json.loads(path.read_bytes())
        except (ValueError, UnicodeDecodeError):
            record = None
            integrity.append("raw_invalid_json")
        normalized, reasons, flags = normalize_record(record, provenance, config)
        if collector_inventory is not None:
            source_id = normalized.get("vehicle_id") or path.stem
            discovered = collector_inventory.get(source_id)
            if not isinstance(discovered, dict):
                integrity.append("raw_inventory_missing_identifier")
            else:
                if discovered.get("vehicle_id") != source_id:
                    integrity.append("raw_inventory_id_mismatch")
                if discovered.get("status") != "fetched":
                    integrity.append("raw_inventory_status:" + str(discovered.get("status") or "missing"))
                menu_provenance = discovered.get("provenance")
                if not isinstance(menu_provenance, list) or not menu_provenance or any(not isinstance(item, dict) for item in menu_provenance):
                    integrity.append("raw_inventory_missing_or_invalid_provenance")
                else:
                    identities = {(item.get("model_year"), item.get("manufacturer"), item.get("model_name")) for item in menu_provenance}
                    if len(identities) != 1:
                        integrity.append("raw_conflicting_menu_provenance")
                    actual_identity = (normalized["model_year"], normalized["manufacturer"], normalized["model_name"])
                    if actual_identity not in identities:
                        integrity.append("raw_menu_record_identity_mismatch")
                    if any(any(value is None or value == "" for value in identity) for identity in identities):
                        integrity.append("raw_inventory_missing_or_invalid_provenance")
        if isinstance(record, dict):
            all_raw_records.append(record)
            raw_keys.update(record)
            for key, value in record.items():
                field_types[key][type(value).__name__] += 1
                if value is None or value == "":
                    missing_fields[key] += 1
            if normalized["vehicle_id"] and path.stem != normalized["vehicle_id"]:
                integrity.append("filename_vehicle_id_mismatch")
        reasons = list(dict.fromkeys(integrity + reasons))
        all_reasons.append(reasons)
        row = {"vehicle_id": normalized.get("vehicle_id") or path.stem, "raw_file": relative, "accepted": not reasons, "reasons": "|".join(reasons), "audit_flags": "|".join(flags)}
        inventory.append(row)
        if reasons:
            exclusions.append(row)
        else:
            accepted.append(normalized)
            raw_by_id[normalized["vehicle_id"]] = record
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in accepted:
        groups[_fingerprint([row[field] for field in CANDIDATE_FIELDS])].append(row)
    candidates: list[dict] = []
    resolution: list[dict] = []
    drop_ids: set[str] = set()
    unresolved_groups = 0
    for candidate_key, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda row: int(row["vehicle_id"]))
        exact_groups: dict[str, list[dict]] = defaultdict(list)
        for row in members:
            exact_groups[_fingerprint(_complete_comparable_record(raw_by_id[row["vehicle_id"]]))].append(row)
        target_conflict = len({row["combined_mpg"] for row in members}) > 1
        unresolved = len(exact_groups) > 1
        unresolved_groups += int(unresolved)
        for row in members:
            candidates.append({"candidate_group": candidate_key, "vehicle_id": row["vehicle_id"], "candidate_size": len(members), "target_conflict": target_conflict, "status": "retained_raw_differences_unresolved" if unresolved else "confirmed_complete_source_equivalence"})
        for equal_members in exact_groups.values():
            representative = equal_members[0]["vehicle_id"]
            for row in equal_members:
                removed = row["vehicle_id"] != representative
                if removed:
                    drop_ids.add(row["vehicle_id"])
                resolution.append({"candidate_group": candidate_key, "vehicle_id": row["vehicle_id"], "representative_id": representative, "removed": removed, "decision": "complete_source_record_equivalence" if len(equal_members) > 1 else "retain_raw_differences", "target_conflict_in_candidate_group": target_conflict})
                if unresolved:
                    extra = "duplicate_candidate_unresolved" + ("|duplicate_candidate_target_conflict" if target_conflict else "")
                    row["audit_flags"] = "|".join(filter(None, [row["audit_flags"], extra]))
    final_rows = [row for row in accepted if row["vehicle_id"] not in drop_ids]
    table = pd.DataFrame(final_rows, columns=NORMALIZED_COLUMNS)
    for column in ("model_year", "cylinders"):
        table[column] = pd.array(table[column], dtype="Int64")
    for column in ("displacement_l", "combined_mpg", "combined_mpg_unrounded", "target_l100km"):
        table[column] = pd.to_numeric(table[column], errors="raise").astype(float)
    table.to_csv(processed / "vehicles.csv", index=False)
    table.to_parquet(processed / "vehicles.parquet", index=False)
    pd.DataFrame(inventory).to_csv(interim / "vehicle_inventory.csv", index=False)
    pd.DataFrame(exclusions, columns=["vehicle_id", "raw_file", "accepted", "reasons", "audit_flags"]).to_csv(interim / "excluded_records.csv", index=False)
    pd.DataFrame(candidates, columns=["candidate_group", "vehicle_id", "candidate_size", "target_conflict", "status"]).to_csv(interim / "duplicate_candidates.csv", index=False)
    pd.DataFrame(resolution, columns=["candidate_group", "vehicle_id", "representative_id", "removed", "decision", "target_conflict_in_candidate_group"]).to_csv(interim / "duplicate_resolution.csv", index=False)
    stages = []
    remaining = list(range(len(all_reasons)))
    for stage in FILTER_ORDER:
        removed = [index for index in remaining if any(_stage(reason) == stage for reason in all_reasons[index])]
        stages.append({"stage": stage, "before": len(remaining), "removed": len(removed), "after": len(remaining) - len(removed)})
        remaining = [index for index in remaining if index not in set(removed)]
    stages.append({"stage": "confirmed_complete_source_duplicates", "before": len(accepted), "removed": len(drop_ids), "after": len(final_rows)})
    pd.DataFrame(stages).to_csv(interim / "cleaning_summary.csv", index=False)
    categories = []
    for column in config["features"]["categorical"] + ["fuel_type_primary", "atv_type"]:
        for value, count in table[column].fillna("<MISSING>").value_counts(dropna=False).items():
            categories.append({"field": column, "value": value, "count": int(count), "scope": "included_rows"})
    # Include observed excluded categories so new API classes remain visible.
    raw_inventory = Counter((str(raw.get("VClass") or "<MISSING>"), str(raw.get("atvType") or raw.get("atvtype") or "<EMPTY>")) for raw in all_raw_records)
    for (vclass, atv), count in raw_inventory.items():
        categories.append({"field": "vehicle_class_atv_pair", "value": vclass + "::" + atv, "count": count, "scope": "all_parseable_raw_records_including_excluded"})
    for field in ("VClass", "fuelType1", "fuelType2", "atvType", "evMotor", "make"):
        observed = Counter(str(raw.get(field) if raw.get(field) is not None else "<MISSING>") or "<EMPTY>" for raw in all_raw_records)
        for value, count in sorted(observed.items()):
            categories.append({"field": field, "value": value, "count": count, "scope": "all_parseable_raw_records_including_excluded"})
    pd.DataFrame(categories, columns=["field", "value", "count", "scope"]).to_csv(interim / "category_inventory.csv", index=False)
    dictionary = {key: {"source": "FuelEconomy.gov individual vehicle JSON", "observed_json_types": dict(field_types[key]), "present_count": sum(field_types[key].values()), "missing_or_empty_count": len(all_raw_records) - sum(field_types[key].values()) + missing_fields[key], "description": f"Raw source field {key}; complete source meaning still requires data-dictionary review.", "meaning_review_status": "unreviewed_raw_field", "internal_field": next((internal for internal, api in FIELD_MAP.items() if api == key), None), "disposition": _disposition(key, config)} for key in sorted(raw_keys)}
    write_json(interim / "raw_field_dictionary.json", dictionary)
    reason_counts = Counter(reason for reasons in all_reasons for reason in reasons)
    summary = {
        "snapshot_id": snapshot_id, "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_vehicle_files": len(files), "accepted_before_duplicate_resolution": len(accepted),
        "collector_inventory_validation": "checked" if collector_inventory is not None else "not_available_legacy_or_test_fixture",
        "excluded_records": len(exclusions), "confirmed_duplicate_records_removed": len(drop_ids),
        "unresolved_candidate_groups": unresolved_groups, "cleaned_rows": len(table),
        "unique_vehicle_ids": int(table["vehicle_id"].nunique()),
        "minimum_distinct_rows": scope_minimum(config),
        "minimum_rows_met": len(table) >= scope_minimum(config),
        "desired_rows_met": len(table) >= config["scope"]["desired_distinct_rows"],
        "distinct_count_status": "provisional_unresolved_duplicate_candidates" if unresolved_groups else "source_configurations_after_confirmed_duplicates",
        "independent_reason_counts": dict(reason_counts),
        "category_allowlist_status": config["scope"].get("class_allowlist_status"),
        "processed_csv_sha256": sha256_file(processed / "vehicles.csv"),
        "processed_parquet_sha256": sha256_file(processed / "vehicles.parquet"),
        "config_sha256": _fingerprint(config),
        "cleaning_module_sha256": sha256_file(Path(__file__)),
    }
    write_json(interim / "cleaned_summary.json", summary)
    roles = {key: "audit_only" for key in NORMALIZED_COLUMNS}
    for key in config["features"]["numeric"] + config["features"]["categorical"]:
        roles[key] = "structured_predictor"
    for key in config["features"]["final_text"]:
        roles[key] = "original_text_final_only_requires_sanitization"
    roles["target_l100km"] = "target"
    roles["model_group"] = "split_group_never_predictor"
    write_json(existing_schema, {"schema_version": config["schema_version"], "snapshot_id": snapshot_id, "target_unit": "L/100 km", "target_formula": "235.2145833333333 / comb08", "columns": [{"name": key, "dtype": str(table[key].dtype), "role": roles[key], "api_field": FIELD_MAP.get(key)} for key in NORMALIZED_COLUMNS], "structured_feature_allowlist": config["features"]["numeric"] + config["features"]["categorical"], "source_config": config, "artifact_hashes": {"vehicles.csv": summary["processed_csv_sha256"], "vehicles.parquet": summary["processed_parquet_sha256"]}})
    return summary


def scope_minimum(config: dict) -> int:
    return int(config["scope"]["minimum_distinct_rows"])


def resolve_snapshot(value: str | Path, root: Path | None = None) -> Path:
    """Accept an explicit existing path, otherwise a snapshot name under data/raw."""
    path = Path(value)
    root = Path(root) if root is not None else project_root()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    if (root / path).exists():
        return (root / path).resolve()
    return root / "data" / "raw" / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a collected snapshot with source integrity and exclusion audits.")
    parser.add_argument("--config", default="configs/project.json")
    parser.add_argument("--snapshot", required=True, help="Snapshot directory or name under data/raw.")
    parser.add_argument("--output-root", help="Separate project-style directory for a development dataset and its audits.")
    parser.add_argument("--allow-small", action="store_true", help="Permit development snapshots below the course minimum; status remains explicit.")
    args = parser.parse_args()
    root = project_root()
    snapshot = resolve_snapshot(args.snapshot, root)
    output_root = Path(args.output_root).resolve() if args.output_root else root
    summary = build_dataset(snapshot, output_root, load_config(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["minimum_rows_met"] and not args.allow_small:
        raise SystemExit("Dataset is below minimum_distinct_rows. Audits were written; use --allow-small only for development.")


if __name__ == "__main__":
    main()
