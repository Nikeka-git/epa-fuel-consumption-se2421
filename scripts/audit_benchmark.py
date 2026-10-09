"""Read-only benchmark audit; never changes source evidence or cleaning decisions.

Checkpoint mode reads only successful immutable paths named in a single captured
cache_index.json. Final mode additionally reconciles the stopped snapshot's
manifest/inventory and an optional cleaned table. No target-dependent pruning,
alias guessing, deduplication, or split assignment is performed by this script.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from fuel_consumption.api import SchemaError, parse_menu, validate_vehicle
from fuel_consumption.clean import CANDIDATE_FIELDS, FIELD_MAP, normalize_record, resolve_snapshot
from fuel_consumption.utils import project_root, write_json


RAW_CATEGORIES = ("year", "make", "VClass", "fuelType1", "fuelType2", "atvType", "atvtype",
                  "evMotor", "phevBlended", "trany", "drive", "baseModel", "basemodel", "startStop")
POWERTRAIN_KEYS = ("fuelType1", "fuelType2", "atvType", "evMotor", "phevBlended")
EXPECTED_X = ("model_year", "displacement_l", "cylinders", "manufacturer", "transmission", "drivetrain", "vehicle_class")
IGNORED_DUPLICATE_KEYS = {"id", "createdOn", "modifiedOn"}


def digest_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def group_text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else " ".join(str(value).casefold().split())


def safe_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Missing response filename")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Response path is not relative to snapshot")
    resolved = (root / relative).resolve()
    resolved.relative_to(root.resolve())
    return resolved


def target_status(record: dict) -> str:
    if "comb08" not in record:
        return "key_absent"
    value = record["comb08"]
    if value is None:
        return "null"
    if isinstance(value, str) and not value.strip():
        return "empty"
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return "invalid_type"
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return "non_numeric"
    if not math.isfinite(value):
        return "nonfinite"
    return "valid_positive_finite" if value > 0 else "nonpositive"


def complete_comparable(record: dict) -> dict:
    """Ignore documented identity/date-only metadata; retain every other value."""
    comparable = {key: value for key, value in record.items() if key not in IGNORED_DUPLICATE_KEYS}
    emissions = comparable.get("emissionsList")
    if isinstance(emissions, dict):
        emissions = dict(emissions)
        details = emissions.get("emissionsInfo")
        if isinstance(details, list):
            emissions["emissionsInfo"] = [{key: value for key, value in row.items() if key != "id"}
                                         if isinstance(row, dict) else row for row in details]
        elif isinstance(details, dict):
            emissions["emissionsInfo"] = {key: value for key, value in details.items() if key != "id"}
        comparable["emissionsList"] = emissions
    return comparable


def candidate_audit(eligible: list[dict], raw_by_id: dict[str, dict]) -> tuple[list[dict], list[dict], dict]:
    """Form candidates solely from the contract's technical identity fields."""
    forbidden = {"combined_mpg", "combined_mpg_unrounded", "target_l100km"}
    if forbidden.intersection(CANDIDATE_FIELDS):
        raise ValueError("Candidate formation unexpectedly contains target/output fields")
    groups = defaultdict(list)
    for row in eligible:
        signature = digest_bytes(canonical([row[field] for field in CANDIDATE_FIELDS]).encode())
        groups[signature].append(row)
    member_rows, difference_rows = [], []
    equivalent_removed, unresolved, conflict_groups = 0, 0, 0
    for signature, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda row: int(row["vehicle_id"]))
        representative = members[0]["vehicle_id"]
        reference = complete_comparable(raw_by_id[representative])
        comparable = {row["vehicle_id"]: complete_comparable(raw_by_id[row["vehicle_id"]]) for row in members}
        complete_signatures = Counter(digest_bytes(canonical(value).encode()) for value in comparable.values())
        target_conflict = len({row["combined_mpg"] for row in members}) > 1
        has_differences = len(complete_signatures) > 1
        equivalent_removed += sum(size - 1 for size in complete_signatures.values())
        unresolved += int(has_differences)
        conflict_groups += int(target_conflict)
        for row in members:
            vehicle_id = row["vehicle_id"]
            source = comparable[vehicle_id]
            differences = []
            for key in sorted(set(reference) | set(source)):
                before, after = reference.get(key), source.get(key)
                if (key in reference) != (key in source) or before != after:
                    differences.append(key)
                    difference_rows.append({"candidate_group": signature, "reference_id": representative,
                                            "vehicle_id": vehicle_id, "raw_field": key,
                                            "reference_key_present": key in reference, "member_key_present": key in source,
                                            "reference_value_json": canonical(before), "member_value_json": canonical(after)})
            member_rows.append({"candidate_group": signature, "vehicle_id": vehicle_id, "reference_id": representative,
                                "n_members": len(members), "differing_raw_keys": " | ".join(differences),
                                "complete_equivalence_sha256": digest_bytes(canonical(source).encode()),
                                "target_conflict_observed_after_formation": target_conflict,
                                "decision_hint": "review_raw_differences_without_automatic_merge" if has_differences
                                                 else "complete_raw_equivalence_except_identity_dates"})
    return member_rows, difference_rows, {"candidate_formation_fields": CANDIDATE_FIELDS,
                                         "target_used_for_candidate_formation": False,
                                         "candidate_groups": len({row["candidate_group"] for row in member_rows}),
                                         "candidate_rows": len(member_rows), "raw_difference_groups": unresolved,
                                         "target_conflict_groups": conflict_groups,
                                         "complete_equivalence_duplicate_rows": equivalent_removed,
                                         "scope": "scope_eligible_before_duplicate_resolution"}


def family_audit(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    base_inventory = defaultdict(set)
    model_inventory = defaultdict(list)
    for row in rows:
        make = group_text(row["manufacturer"])
        base = group_text(row["base_model"])
        if base:
            base_inventory[make].add(base)
        model_inventory[(make, group_text(row["model_name"]))].append(row)
    fallbacks, consistency = [], []
    for row in rows:
        if group_text(row["base_model"]):
            continue
        make, model = group_text(row["manufacturer"]), group_text(row["model_name"])
        fallbacks.append({"vehicle_id": row["vehicle_id"], "manufacturer": row["manufacturer"],
                          "model_name": row["model_name"], "model_year": row["model_year"],
                          "status": "exact_normalized_name_matches_observed_base_model" if model in base_inventory[make]
                                    else "unresolved_fallback_requires_source_review",
                          "automatic_alias_change": False})
    for (make, model), members in sorted(model_inventory.items()):
        bases = sorted({group_text(row["base_model"]) for row in members if group_text(row["base_model"])})
        status = "multiple_base_models_for_same_full_model_review" if len(bases) > 1 else "consistent_observed_base_model"
        if not bases:
            status = "all_base_models_missing_fallback_review"
        consistency.append({"normalized_manufacturer": make, "normalized_full_model_name": model,
                            "n_rows": len(members), "n_base_models": len(bases), "base_models": " | ".join(bases),
                            "years": " | ".join(str(year) for year in sorted({row["model_year"] for row in members})),
                            "missing_base_rows": sum(not group_text(row["base_model"]) for row in members), "status": status})
    return fallbacks, consistency, {"fallback_rows": len(fallbacks),
                                    "unresolved_fallback_rows": sum(row["status"].startswith("unresolved") for row in fallbacks),
                                    "model_names_with_multiple_base_models": sum(row["n_base_models"] > 1 for row in consistency),
                                    "automatic_aliases_applied": 0,
                                    "prefix_alias_guessing": False}


def write_csv(output: Path, name: str, rows: list[dict], columns: list[str]) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(output / name, index=False)


def run_audit(snapshot: Path, output: Path, *, checkpoint: bool = False,
              dataset: Path | None = None, dictionary: Path | None = None) -> dict:
    snapshot, output = snapshot.resolve(), output.resolve()
    if output == snapshot or snapshot in output.parents:
        raise ValueError("Audit output must be outside the immutable source snapshot")
    if not checkpoint and (snapshot / ".collector.lock").exists():
        raise ValueError("Final audit requires a stopped snapshot; use --checkpoint while collection is active")
    config_body = (snapshot / "config.json").read_bytes()
    config = json.loads(config_body)
    cache_body = (snapshot / "cache_index.json").read_bytes()
    cache = json.loads(cache_body)
    entries = cache.get("entries", cache)
    if not isinstance(entries, dict):
        raise ValueError("cache_index must contain a mapping")
    inventory_body = (snapshot / "inventory.json").read_bytes()
    inventory = json.loads(inventory_body).get("records")
    if not isinstance(inventory, dict):
        raise ValueError("inventory must contain a records mapping")
    metadata_body = (snapshot / "snapshot.json").read_bytes()
    metadata = json.loads(metadata_body)
    snapshot_id = metadata["snapshot_id"]
    expected_x = config["features"]["numeric"] + config["features"]["categorical"]
    if set(expected_x) != set(EXPECTED_X):
        raise ValueError("Structured X allowlist differs from the seven-field project contract")
    raw_by_id, normalized_by_id, eligible = {}, {}, []
    integrity, record_rows, missing_metadata = [], [], []
    category_counts, target_counts, reason_counts, raw_keys = Counter(), Counter(), Counter(), set()
    field_types, field_missing = defaultdict(Counter), Counter()
    for url, entry in sorted(entries.items()):
        problems, raw, payload = [], None, None
        try:
            path = safe_path(snapshot, entry.get("file"))
            raw = path.read_bytes()
            if digest_bytes(raw) != entry.get("sha256"):
                problems.append("checksum_mismatch")
            if entry.get("status") != 200 or entry.get("valid") is not True:
                problems.append("cached_entry_not_valid_http_200")
            if entry.get("url") != url or not entry.get("received_at_utc"):
                problems.append("incomplete_or_conflicting_response_provenance")
            payload = json.loads(raw)
            if entry.get("kind") == "menu":
                parse_menu(payload)
            elif entry.get("kind") == "vehicle":
                validate_vehicle(payload, **entry.get("expected", {}))
            else:
                problems.append("unknown_resource_kind")
        except (OSError, ValueError, TypeError, SchemaError) as exc:
            problems.append(f"invalid_cached_evidence:{type(exc).__name__}:{exc}")
        vehicle_id = str(payload.get("id")) if isinstance(payload, dict) and entry.get("kind") == "vehicle" else None
        if vehicle_id is not None:
            discovered = inventory.get(vehicle_id)
            if not isinstance(discovered, dict):
                problems.append("vehicle_missing_from_inventory")
            else:
                provenance = discovered.get("provenance", [])
                if not provenance or any(not isinstance(row, dict) for row in provenance):
                    problems.append("invalid_menu_provenance")
                else:
                    identities = {(row.get("model_year"), row.get("manufacturer"), row.get("model_name")) for row in provenance}
                    try:
                        expected_identity = (int(payload["year"]), payload["make"], payload["model"])
                    except (ValueError, TypeError, KeyError):
                        expected_identity = None
                    if expected_identity is None or identities != {expected_identity}:
                        problems.append("conflicting_menu_provenance")
                if discovered.get("status") != "fetched":
                    problems.append("checkpoint_inventory_lag" if checkpoint else "inventory_not_fetched")
        integrity.append({"url": url, "kind": entry.get("kind"), "file": entry.get("file"),
                          "vehicle_id": vehicle_id, "expected_sha256": entry.get("sha256"),
                          "observed_sha256": digest_bytes(raw) if raw is not None else None,
                          "status": "issues" if problems else "verified", "issues": " | ".join(problems)})
        blocking = [issue for issue in problems if issue != "checkpoint_inventory_lag"]
        if entry.get("kind") != "vehicle" or blocking or not isinstance(payload, dict):
            continue
        provenance = dict(entry, snapshot_id=snapshot_id)
        normalized, reasons, flags = normalize_record(payload, provenance, config)
        raw_by_id[vehicle_id], normalized_by_id[vehicle_id] = payload, normalized
        if not reasons:
            eligible.append(normalized)
        record_rows.append({"vehicle_id": vehicle_id, "scope_eligible": not reasons,
                            "reasons": " | ".join(reasons), "audit_flags": " | ".join(flags),
                            "target_status": target_status(payload)})
        reason_counts.update(reasons)
        target_counts[target_status(payload)] += 1
        raw_keys.update(payload)
        for key, value in payload.items():
            field_types[key][type(value).__name__] += 1
            if value is None or (isinstance(value, str) and not value.strip()):
                field_missing[key] += 1
        for key in RAW_CATEGORIES:
            value = payload.get(key)
            if key not in payload:
                label, raw_json = "<KEY_ABSENT>", "null"
            elif value is None:
                label, raw_json = "<NULL>", "null"
            elif isinstance(value, str) and not value.strip():
                label, raw_json = "<EMPTY>", canonical(value)
            else:
                label, raw_json = str(value), canonical(value)
            category_counts[("verified_committed_raw", key, label, raw_json)] += 1
        for key in POWERTRAIN_KEYS:
            keys = ("atvType", "atvtype") if key == "atvType" else (key,)
            present = any(name in payload for name in keys)
            value = next((payload[name] for name in keys if name in payload), None)
            status = "key_absent" if not present else "null_value" if value is None else "present"
            missing_metadata.append({"vehicle_id": vehicle_id, "metadata_field": key, "status": status})
    fetched_inventory_ids = {vehicle_id for vehicle_id, row in inventory.items() if row.get("status") == "fetched"}
    inventory_only = sorted(fetched_inventory_ids - set(raw_by_id), key=int)
    manifest_summary = {"checked": False, "reason": "checkpoint_reads_committed_cache_only"}
    if not checkpoint:
        manifest_body = (snapshot / "requests.jsonl").read_bytes()
        manifest = [json.loads(line) for line in manifest_body.splitlines() if line.strip()]
        manifest_success = {row["request_id"]: row for row in manifest if row.get("valid") is True and row.get("status") == 200}
        missing_request_ids = [entry.get("request_id") for entry in entries.values()
                               if entry.get("request_id") not in manifest_success]
        mismatch_ids = []
        for entry in entries.values():
            matching = manifest_success.get(entry.get("request_id"))
            if matching and any(matching.get(field) != entry.get(field) for field in ("url", "file", "sha256", "status", "kind", "expected")):
                mismatch_ids.append(entry["request_id"])
        manifest_summary = {"checked": True, "sha256": digest_bytes(manifest_body), "attempts": len(manifest),
                            "successful_responses": len(manifest_success), "failed_attempts": len(manifest) - len(manifest_success),
                            "cached_request_ids_missing_from_successful_manifest": missing_request_ids,
                            "cached_entries_disagreeing_with_manifest": mismatch_ids,
                            "successful_manifest_urls_missing_from_cache": sorted({row["url"] for row in manifest_success.values()} - set(entries))}
    candidates, differences, duplicate_summary = candidate_audit(eligible, raw_by_id)
    family_rows = eligible
    cleaned_summary = {"checked": False, "reason": "no_cleaned_dataset_supplied"}
    cleaned_issues, missingness = [], []
    if dataset is not None:
        dataset_body = dataset.read_bytes()
        frame = pd.read_parquet(dataset) if dataset.suffix == ".parquet" else pd.read_csv(dataset, dtype={"vehicle_id": str})
        frame["vehicle_id"] = frame["vehicle_id"].astype(str)
        if not set(EXPECTED_X).issubset(frame):
            raise ValueError("Cleaned dataset lacks required X fields")
        eligible_ids = {item["vehicle_id"] for item in eligible}
        for row in frame.to_dict("records"):
            vehicle_id = row["vehicle_id"]
            problems = []
            expected = normalized_by_id.get(vehicle_id)
            if expected is None:
                problems.append("cleaned_id_missing_from_verified_raw")
            else:
                if vehicle_id not in eligible_ids:
                    problems.append("cleaned_id_does_not_pass_scope_rules")
                if row.get("snapshot_id") != snapshot_id:
                    problems.append("cleaned_snapshot_id_mismatch")
                if row.get("raw_sha256") != expected["raw_sha256"]:
                    problems.append("cleaned_raw_sha256_mismatch")
                value = row.get("target_l100km")
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    problems.append("cleaned_target_not_positive_finite")
                elif expected["target_l100km"] is None or not math.isclose(value, expected["target_l100km"], rel_tol=1e-12, abs_tol=1e-12):
                    problems.append("cleaned_target_formula_mismatch")
                for field in EXPECTED_X:
                    actual, original = row[field], expected[field]
                    if pd.isna(actual) and original is None:
                        continue
                    if pd.isna(actual) or original is None or actual != original:
                        problems.append("cleaned_x_raw_mismatch:" + field)
            if problems:
                cleaned_issues.append({"vehicle_id": vehicle_id, "issues": " | ".join(problems)})
        for field in EXPECTED_X:
            count = int(frame[field].isna().sum() + frame[field].map(lambda value: isinstance(value, str) and not value.strip()).sum())
            missingness.append({"scope": "cleaned_rows", "field": field, "n_rows": len(frame), "missing_count": count,
                                "missing_fraction": count / len(frame) if len(frame) else None})
        family_rows = frame.to_dict("records")
        cleaned_summary = {"checked": True, "path": dataset.as_posix(), "sha256": digest_bytes(dataset_body), "rows": len(frame),
                           "unique_vehicle_ids": int(frame["vehicle_id"].nunique()),
                           "duplicate_id_rows": int(frame["vehicle_id"].duplicated().sum()), "issue_rows": len(cleaned_issues),
                           "minimum_rows_met": len(frame) >= config["scope"]["minimum_distinct_rows"],
                           "desired_rows_met": len(frame) >= config["scope"]["desired_distinct_rows"]}
        for field in ("model_year", "manufacturer", "vehicle_class"):
            for value, count in frame[field].value_counts(dropna=False).items():
                category_counts[("cleaned_rows", field, "<MISSING>" if pd.isna(value) else str(value),
                                 "null" if pd.isna(value) else canonical(value.item() if hasattr(value, "item") else value))] += int(count)
    for field in EXPECTED_X:
        count = sum(row[field] is None for row in eligible)
        missingness.append({"scope": "scope_eligible_before_duplicates", "field": field, "n_rows": len(eligible),
                            "missing_count": count, "missing_fraction": count / len(eligible) if eligible else None})
    fallbacks, family_consistency, family_summary = family_audit(family_rows)
    meaning_dictionary = json.loads(dictionary.read_bytes()) if dictionary is not None else {}
    observations = []
    for key in sorted(raw_keys):
        description = meaning_dictionary.get(key, {})
        observations.append({"raw_field": key, "observed_types_json": canonical(dict(field_types[key])),
                             "present_count": sum(field_types[key].values()),
                             "missing_or_empty_count": len(raw_by_id) - sum(field_types[key].values()) + field_missing[key],
                             "internal_field": next((name for name, api in FIELD_MAP.items() if api == key), ""),
                             "meaning_review_status": description.get("meaning_review_status", "not_verified_by_independent_audit"),
                             "provided_description": description.get("description", ""),
                             "audit_claims_source_meaning_verified": False})
    powertrain_counts = Counter((row["metadata_field"], row["status"]) for row in missing_metadata)
    issue_rows = [row for row in integrity if row["issues"] and row["issues"] != "checkpoint_inventory_lag"]
    hints = []
    if checkpoint:
        hints.append("Checkpoint is a committed-cache view during collection; inventory/cache may be captured at different instants. Counts are provisional.")
    if issue_rows:
        hints.append(f"Inspect {len(issue_rows)} immutable cache/inventory integrity issues before freezing; do not overwrite raw evidence.")
    if inventory_only:
        hints.append(f"{len(inventory_only)} inventory fetched IDs were not verified in the captured cache; checkpoint lag is possible only in checkpoint mode.")
    if manifest_summary.get("cached_request_ids_missing_from_successful_manifest") or manifest_summary.get("cached_entries_disagreeing_with_manifest") or manifest_summary.get("successful_manifest_urls_missing_from_cache"):
        hints.append("Reconcile manifest/cache disagreements before freezing. Recorded request IDs, paths, hashes, and identity expectations must agree.")
    if cleaned_issues or cleaned_summary.get("duplicate_id_rows"):
        hints.append("Reconcile cleaned_integrity_issues.csv and duplicate cleaned IDs with immutable source records before freezing.")
    if duplicate_summary["raw_difference_groups"]:
        hints.append(f"Review {duplicate_summary['raw_difference_groups']} duplicate candidate groups using duplicate_raw_differences.csv. Target did not form candidates; no merge was applied.")
    if duplicate_summary["target_conflict_groups"]:
        hints.append("Candidate groups have different published targets; inspect source specifications and retain/quarantine with a documented decision. Never average conflicting targets silently.")
    if family_summary["unresolved_fallback_rows"] or family_summary["model_names_with_multiple_base_models"]:
        hints.append("Review model_family_fallbacks.csv and model_family_consistency.csv from source evidence before freeze; do not infer aliases by model-name prefixes.")
    if any(key[1] != "present" for key in powertrain_counts):
        hints.append("Missing/null powertrain metadata is not proof of a non-hybrid configuration. Current cleaning rules quarantine these records.")
    if any(row["meaning_review_status"] in {"not_verified_by_independent_audit", "unreviewed_raw_field"} for row in observations):
        hints.append("Detailed meanings remain unverified for some raw fields. Observed types/roles are not substitutes for the official source dictionary.")
    hints.extend(["No target outlier pruning or target imputation was performed. Counts describe catalogue configurations, not vehicle sales.",
                  "Unknown/non-allowed classes and alternative-technology labels are reported; this audit does not extend the allowlist."])
    summary = {"audit_version": "1", "audited_at_utc": datetime.now(timezone.utc).isoformat(), "snapshot_id": snapshot_id,
               "mode": "checkpoint" if checkpoint else "final_stopped_snapshot", "collection_status": metadata.get("status"),
               "full_catalogue_complete": metadata.get("full_catalogue_complete", False),
               "captured_source_sha256": {"config.json": digest_bytes(config_body), "cache_index.json": digest_bytes(cache_body),
                                          "inventory.json": digest_bytes(inventory_body), "snapshot.json": digest_bytes(metadata_body)},
               "committed_successful_cache_entries": len(entries), "verified_vehicle_records": len(raw_by_id),
               "scope_eligible_before_duplicate_resolution": len(eligible), "integrity_issue_rows": len(issue_rows),
               "inventory_fetched_ids_not_verified_in_captured_cache": inventory_only,
               "inventory_status_counts": dict(Counter(row.get("status", "missing") for row in inventory.values())),
               "target_validity_counts": dict(target_counts), "independent_exclusion_reason_counts": dict(reason_counts),
               "missing_powertrain_counts": {f"{field}:{status}": count for (field, status), count in sorted(powertrain_counts.items())},
               "manifest": manifest_summary, "duplicates": duplicate_summary, "families": family_summary, "cleaned": cleaned_summary,
               "raw_field_count": len(observations), "meaning_review_status_counts": dict(Counter(row["meaning_review_status"] for row in observations)),
               "human_review_hints": hints, "source_data_or_aliases_modified": False, "target_outlier_pruning": False}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "audit_summary.json", summary)
    write_csv(output, "raw_integrity.csv", integrity, ["url", "kind", "file", "vehicle_id", "expected_sha256", "observed_sha256", "status", "issues"])
    write_csv(output, "raw_scope_decisions.csv", record_rows, ["vehicle_id", "scope_eligible", "reasons", "audit_flags", "target_status"])
    categories = [{"scope": scope, "field": field, "value": value, "raw_value_json": raw_json, "count": count}
                  for (scope, field, value, raw_json), count in sorted(category_counts.items())]
    write_csv(output, "category_counts.csv", categories, ["scope", "field", "value", "raw_value_json", "count"])
    write_csv(output, "powertrain_metadata_presence.csv", missing_metadata, ["vehicle_id", "metadata_field", "status"])
    write_csv(output, "duplicate_candidates.csv", candidates, ["candidate_group", "vehicle_id", "reference_id", "n_members", "differing_raw_keys", "complete_equivalence_sha256", "target_conflict_observed_after_formation", "decision_hint"])
    write_csv(output, "duplicate_raw_differences.csv", differences, ["candidate_group", "reference_id", "vehicle_id", "raw_field", "reference_key_present", "member_key_present", "reference_value_json", "member_value_json"])
    write_csv(output, "model_family_fallbacks.csv", fallbacks, ["vehicle_id", "manufacturer", "model_name", "model_year", "status", "automatic_alias_change"])
    write_csv(output, "model_family_consistency.csv", family_consistency, ["normalized_manufacturer", "normalized_full_model_name", "n_rows", "n_base_models", "base_models", "years", "missing_base_rows", "status"])
    write_csv(output, "x_missingness.csv", missingness, ["scope", "field", "n_rows", "missing_count", "missing_fraction"])
    write_csv(output, "cleaned_integrity_issues.csv", cleaned_issues, ["vehicle_id", "issues"])
    write_csv(output, "raw_field_observations.csv", observations, ["raw_field", "observed_types_json", "present_count", "missing_or_empty_count", "internal_field", "meaning_review_status", "provided_description", "audit_claims_source_meaning_verified"])
    review = (f"# Independent benchmark audit: {snapshot_id}\n\n"
              f"Mode: {summary['mode']}. Verified committed records: {len(raw_by_id)}. Scope-eligible before duplicates: {len(eligible)}. "
              f"Immutable integrity issue rows: {len(issue_rows)}.\n\n"
              + "\n".join(f"- {hint}" for hint in hints) + "\n")
    (output / "REVIEW.md").write_text(review, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="Existing snapshot path or name under data/raw")
    parser.add_argument("--output", type=Path, help="Audit directory outside the raw snapshot")
    parser.add_argument("--checkpoint", action="store_true", help="Read committed cache during active collection; findings are provisional")
    parser.add_argument("--dataset", type=Path, help="Optional cleaned CSV/parquet to reconcile")
    parser.add_argument("--dictionary", type=Path, help="Optional reviewed raw-field dictionary; observed data do not verify its meanings")
    args = parser.parse_args()
    root = project_root()
    snapshot = resolve_snapshot(args.snapshot, root)
    output = args.output or root / "reports" / "audit" / (snapshot.name + ("_checkpoint" if args.checkpoint else ""))
    summary = run_audit(snapshot, output, checkpoint=args.checkpoint, dataset=args.dataset, dictionary=args.dictionary)
    print(json.dumps({key: summary[key] for key in ("snapshot_id", "mode", "verified_vehicle_records", "scope_eligible_before_duplicate_resolution", "integrity_issue_rows", "duplicates", "families", "cleaned")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
