"""Offline domain tests using immutable real API records and modified copies."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from fuel_consumption.clean import build_dataset, normalize_record, resolve_snapshot


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))


def pilot(vehicle_id):
    return json.loads((ROOT / "evidence" / "api_probe" / f"vehicle_{vehicle_id}.json").read_text(encoding="utf-8"))


@pytest.fixture
def provenance():
    return {"url": "https://www.fueleconomy.gov/ws/rest/vehicle/48910", "received_at_utc": "2026-10-09T00:00:00+00:00", "sha256": "a" * 64, "snapshot_id": "offline_fixture"}


@pytest.mark.parametrize("vehicle_id,expected", [(34836, True), (48910, True), (48861, False), (49014, False), (48147, False)])
def test_real_powertrain_scope(vehicle_id, expected, config, provenance):
    row, reasons, _ = normalize_record(pilot(vehicle_id), provenance, config)
    assert (not reasons) is expected
    if expected:
        assert row["target_l100km"] > 0
    else:
        assert "alternative_technology_present" in reasons


def test_us_mpg_conversion_and_start_stop(config, provenance):
    record = pilot(48910)
    assert record["startStop"] == "Y"
    row, reasons, _ = normalize_record(record, provenance, config)
    assert not reasons
    assert row["target_l100km"] == pytest.approx(7.84048611111111, rel=1e-12)


@pytest.mark.parametrize("target", [None, "", "NaN", "inf", "-inf", "0", "-1", "1e-310", {}, True])
def test_bad_target_cannot_enter_dataset(target, config, provenance):
    record = pilot(48910)
    record["comb08"] = target
    row, reasons, _ = normalize_record(record, provenance, config)
    assert "invalid_target" in reasons
    assert row["target_l100km"] is None


@pytest.mark.parametrize("field", ["atvType", "fuelType1", "fuelType2", "evMotor", "phevBlended"])
def test_missing_powertrain_key_is_quarantined(field, config, provenance):
    record = pilot(48910)
    del record[field]
    _, reasons, _ = normalize_record(record, provenance, config)
    assert f"missing_powertrain_metadata:{field}" in reasons


@pytest.mark.parametrize("field", ["atvType", "fuelType2", "evMotor", "phevBlended"])
def test_null_powertrain_is_not_confirmed_absence(field, config, provenance):
    record = pilot(48910)
    record[field] = None
    _, reasons, _ = normalize_record(record, provenance, config)
    assert f"null_powertrain_metadata:{field}" in reasons


def test_compatibility_aliases_are_flagged_and_conflicts_excluded(config, provenance):
    record = pilot(48910)
    record["atvtype"] = record.pop("atvType")
    record["basemodel"] = record.pop("baseModel")
    row, reasons, flags = normalize_record(record, provenance, config)
    assert not reasons
    assert row["model_group"] == "toyota::rav4"
    assert "compatibility_alias:atvtype->atvType" in flags
    record["atvType"] = "Hybrid"
    _, reasons, _ = normalize_record(record, provenance, config)
    assert "conflicting_alias:atvType" in reasons


@pytest.mark.parametrize("field,value", [("displ", "-1"), ("displ", "0"), ("displ", "oops"), ("cylinders", "3.5"), ("cylinders", "0"), ("cylinders", float("inf"))])
def test_bad_engine_spec_is_quarantined(field, value, config, provenance):
    record = pilot(48910)
    record[field] = value
    _, reasons, _ = normalize_record(record, provenance, config)
    assert any(reason.startswith(("invalid_numeric:", "nonpositive_engine_spec:")) for reason in reasons)


def test_missing_numeric_engine_fields_allow_only_one(config, provenance):
    record = pilot(48910)
    record["displ"] = ""
    row, reasons, flags = normalize_record(record, provenance, config)
    assert not reasons
    assert row["displacement_l"] is None
    assert "missing_optional:displacement_l" in flags
    record["cylinders"] = ""
    _, reasons, _ = normalize_record(record, provenance, config)
    assert "insufficient_engine_specs" in reasons


def test_all_reasons_retained_and_hybrid_token_boundaries(config, provenance):
    record = pilot(48910)
    record.update({"model": "Example HEV", "comb08": "0", "fuelType2": "Electricity"})
    _, reasons, _ = normalize_record(record, provenance, config)
    assert {"hybrid_text_signal", "invalid_target", "secondary_fuel_present"}.issubset(reasons)
    record.update({"model": "HEVenture", "comb08": "30", "fuelType2": ""})
    _, reasons, _ = normalize_record(record, provenance, config)
    assert not reasons


def write_snapshot(tmp_path, records):
    snapshot = tmp_path / "raw" / "offline_fixture"
    (snapshot / "vehicles").mkdir(parents=True)
    manifest = []
    for record in records:
        data = json.dumps(record).encode("utf-8")
        filename = f"vehicles/{record['id']}.json"
        (snapshot / filename).write_bytes(data)
        manifest.append({"file": filename, "sha256": hashlib.sha256(data).hexdigest(), "received_at_utc": "2026-10-09T00:00:00+00:00", "url": f"https://www.fueleconomy.gov/ws/rest/vehicle/{record['id']}", "status": 200, "valid": True})
    (snapshot / "requests.jsonl").write_text("\n".join(json.dumps(entry) for entry in manifest), encoding="utf-8")
    return snapshot


def test_build_real_pilot_with_audits_and_leakage_allowlist(tmp_path, config):
    snapshot = write_snapshot(tmp_path, [pilot(vehicle_id) for vehicle_id in [34836, 48910, 48861, 49014, 48147]])
    output_root = tmp_path / "development"
    result = build_dataset(snapshot, output_root, config)
    assert result["raw_vehicle_files"] == 5
    assert result["cleaned_rows"] == 2
    assert result["excluded_records"] == 3
    assert result["minimum_rows_met"] is False
    table = pd.read_parquet(output_root / "data/processed/vehicles.parquet")
    schema = json.loads((output_root / "data/processed/schema.json").read_text(encoding="utf-8"))
    features = schema["structured_feature_allowlist"]
    assert set(features) == {"model_year", "displacement_l", "cylinders", "manufacturer", "transmission", "drivetrain", "vehicle_class"}
    assert all(table["target_l100km"].notna())
    dictionary = json.loads((output_root / "data/interim/raw_field_dictionary.json").read_text(encoding="utf-8"))
    assert set(dictionary) == set().union(*(pilot(vehicle_id).keys() for vehicle_id in [34836, 48910, 48861, 49014, 48147]))
    assert dictionary["co2"]["disposition"] == "excluded_from_predictors_leakage_or_output"
    categories = pd.read_csv(output_root / "data/interim/category_inventory.csv")
    assert any("Hybrid" in str(value) for value in categories["value"])


def test_complete_duplicates_removed_ambiguous_specs_retained(tmp_path, config):
    first = pilot(48910)
    exact = deepcopy(first)
    exact["id"] = "90001"
    for row in exact["emissionsList"]["emissionsInfo"]:
        row["id"] = exact["id"]
    ambiguous = deepcopy(first)
    ambiguous["id"] = "90002"
    ambiguous["comb08"] = "29"
    ambiguous["hpv"] = "100"
    result = build_dataset(write_snapshot(tmp_path, [first, exact, ambiguous]), tmp_path / "development", config)
    assert result["accepted_before_duplicate_resolution"] == 3
    assert result["confirmed_duplicate_records_removed"] == 1
    assert result["cleaned_rows"] == 2
    assert result["unresolved_candidate_groups"] == 1
    table = pd.read_parquet(tmp_path / "development/data/processed/vehicles.parquet")
    assert set(table["vehicle_id"]) == {"48910", "90002"}
    assert all(table["audit_flags"].str.contains("duplicate_candidate_target_conflict"))


def test_source_checksum_mismatch_is_excluded(tmp_path, config):
    snapshot = write_snapshot(tmp_path, [pilot(48910)])
    path = snapshot / "vehicles/48910.json"
    path.write_bytes(path.read_bytes() + b"\n")
    result = build_dataset(snapshot, tmp_path / "development", config)
    assert result["cleaned_rows"] == 0
    assert result["independent_reason_counts"]["raw_checksum_missing_or_mismatch"] == 1


def test_existing_different_snapshot_is_not_overwritten(tmp_path, config):
    output_root = tmp_path / "development"
    snapshot = write_snapshot(tmp_path, [pilot(48910)])
    build_dataset(snapshot, output_root, config)
    other = snapshot.with_name("other_snapshot")
    snapshot.rename(other)
    with pytest.raises(ValueError, match="another snapshot"):
        build_dataset(other, output_root, config)


def test_existing_changed_target_configuration_is_not_overwritten(tmp_path, config):
    output_root = tmp_path / "development"
    snapshot = write_snapshot(tmp_path, [pilot(48910)])
    build_dataset(snapshot, output_root, config)
    changed = deepcopy(config)
    changed["target"]["us_mpg_conversion_constant"] = 282.481
    with pytest.raises(ValueError, match="another configuration"):
        build_dataset(snapshot, output_root, changed)


def test_collector_inventory_conflict_failed_pending_and_missing_ids_quarantined(tmp_path, config):
    records = [pilot(48910)]
    for vehicle_id in ("90001", "90002", "90003"):
        record = deepcopy(records[0])
        record["id"] = vehicle_id
        records.append(record)
    snapshot = write_snapshot(tmp_path, records)
    inventory = {}
    for record in records[:3]:
        inventory[record["id"]] = {
            "vehicle_id": record["id"], "status": "fetched",
            "provenance": [{"model_year": 2025, "manufacturer": "Toyota", "model_name": "RAV4"}],
        }
    inventory["90001"]["provenance"].append({"model_year": 2025, "manufacturer": "Toyota", "model_name": "DifferentModel"})
    inventory["90001"]["status"] = "failed"
    inventory["90002"]["status"] = "discovered"
    (snapshot / "inventory.json").write_text(json.dumps({"schema_version": "1", "records": inventory}), encoding="utf-8")
    result = build_dataset(snapshot, tmp_path / "development", config)
    assert result["cleaned_rows"] == 1
    assert result["collector_inventory_validation"] == "checked"
    assert result["independent_reason_counts"]["raw_conflicting_menu_provenance"] == 1
    assert result["independent_reason_counts"]["raw_inventory_status:failed"] == 1
    assert result["independent_reason_counts"]["raw_inventory_status:discovered"] == 1
    assert result["independent_reason_counts"]["raw_inventory_missing_identifier"] == 1
    table = pd.read_parquet(tmp_path / "development/data/processed/vehicles.parquet")
    assert list(table["vehicle_id"]) == ["48910"]


def test_collector_menu_record_identity_disagreement_quarantined(tmp_path, config):
    snapshot = write_snapshot(tmp_path, [pilot(48910)])
    document = {"records": {"48910": {"vehicle_id": "48910", "status": "fetched", "provenance": [{"model_year": 2025, "manufacturer": "Toyota", "model_name": "DifferentModel"}]}}}
    (snapshot / "inventory.json").write_text(json.dumps(document), encoding="utf-8")
    result = build_dataset(snapshot, tmp_path / "development", config)
    assert result["cleaned_rows"] == 0
    assert result["independent_reason_counts"]["raw_menu_record_identity_mismatch"] == 1


def test_snapshot_resolution_existing_cwd_root_and_bare_name(tmp_path, monkeypatch):
    project = tmp_path / "project"
    explicit = project / "examples/smoke/raw"
    explicit.mkdir(parents=True)
    other_cwd = tmp_path / "other_cwd"
    local = other_cwd / "my_raw"
    local.mkdir(parents=True)
    monkeypatch.chdir(other_cwd)
    assert resolve_snapshot("my_raw", project) == local.resolve()
    assert resolve_snapshot("examples/smoke/raw", project) == explicit.resolve()
    assert resolve_snapshot(explicit, project) == explicit
    assert resolve_snapshot("new_snapshot", project) == project / "data/raw/new_snapshot"


def test_relocated_snapshot_preserves_original_metadata_identity(tmp_path, config):
    original = write_snapshot(tmp_path, [pilot(48910)])
    original = original.rename(original.with_name("smoke_20261009"))
    (original / "snapshot.json").write_text(json.dumps({"snapshot_id": "smoke_20261009", "status": "partial"}), encoding="utf-8")
    relocated = tmp_path / "examples/smoke/raw"
    shutil.copytree(original, relocated)
    original_result = build_dataset(original, tmp_path / "original_outputs", config)
    relocated_result = build_dataset(relocated, tmp_path / "relocated_outputs", config)
    assert original_result["snapshot_id"] == relocated_result["snapshot_id"] == "smoke_20261009"
    original_table = pd.read_parquet(tmp_path / "original_outputs/data/processed/vehicles.parquet")
    relocated_table = pd.read_parquet(tmp_path / "relocated_outputs/data/processed/vehicles.parquet")
    pd.testing.assert_frame_equal(original_table, relocated_table)
    assert set(relocated_table["snapshot_id"]) == {"smoke_20261009"}
    schema = json.loads((tmp_path / "relocated_outputs/data/processed/schema.json").read_text(encoding="utf-8"))
    assert schema["snapshot_id"] == "smoke_20261009"


@pytest.mark.parametrize("declared_id", [None, "", "../other_snapshot", 123])
def test_invalid_declared_snapshot_identity_not_silently_replaced(tmp_path, config, declared_id):
    snapshot = write_snapshot(tmp_path, [pilot(48910)])
    (snapshot / "snapshot.json").write_text(json.dumps({"snapshot_id": declared_id}), encoding="utf-8")
    with pytest.raises(ValueError, match="valid snapshot_id"):
        build_dataset(snapshot, tmp_path / "development", config)
