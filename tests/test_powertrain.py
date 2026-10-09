"""Confirmed source label omissions, scope boundaries and consistent inference."""
import hashlib
import json
from pathlib import Path

import pytest

from fuel_consumption.clean import normalize_record
from fuel_consumption.features import MIDTERM_FEATURES
from fuel_consumption.powertrain import reviewed_hybrid_rules, reviewed_uncertain_rules
from fuel_consumption.predict import validate_specifications

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/powertrain"


@pytest.mark.parametrize("identifier", ["44187", "44205"])
def test_real_mild_hybrid_without_epa_signals_is_excluded(identifier):
    source = next(row for row in json.loads((FIXTURES / "source_manifest.json").read_text(encoding="utf-8"))
                  if Path(row["file"]).stem == identifier)
    path = FIXTURES / f"{identifier}.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["atvType"] == record["evMotor"] == record["fuelType2"] == ""
    config = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8"))
    row, reasons, _ = normalize_record(record, source, config)
    assert "reviewed_powertrain_hybrid:volvo_b_badge_mild_hybrid" in reasons
    assert row["target_l100km"] > 0
    specifications = {field: row[field] for field in (*MIDTERM_FEATURES, "model_name", "engine_description")}
    with pytest.raises(ValueError, match="reviewed hybrid powertrain"):
        validate_specifications(specifications, config, True)


@pytest.mark.parametrize("make,model,year", [
    ("Ford", "B5", 2022), ("Volvo", "XC40 T5 AWD", 2022),
    ("Volvo", "S60 B50 AWD", 2022), ("Volvo", "S60 B5 AWD", 2021),
])
def test_reviewed_badge_rule_is_bounded_and_manufacturer_specific(make, model, year):
    assert reviewed_hybrid_rules(make, model, year) == []


def test_source_badge_decision_does_not_depend_on_target():
    config = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8"))
    record = json.loads((FIXTURES / "44187.json").read_text(encoding="utf-8"))
    provenance = {"sha256": "a" * 64, "url": "offline-modified-fixture"}
    for mpg in ["1", "100"]:
        _, reasons, _ = normalize_record({**record, "comb08": mpg}, provenance, config)
        assert "reviewed_powertrain_hybrid:volvo_b_badge_mild_hybrid" in reasons


@pytest.mark.parametrize("model", ["A4 quattro", "A4 allroad quattro", "A5 Cabriolet quattro", "Q5 quattro"])
def test_audi_us_2l_2021_scope(model):
    assert reviewed_hybrid_rules("Audi", model, 2021, 2.0, 4) == ["audi_2021_2022_a4_a5_q5_2l_mild_hybrid"]


@pytest.mark.parametrize("model,year,displacement,cylinders", [
    ("A5 Cabriolet quattro", 2020, 2.0, 4), ("A5 quattro", 2023, 2.0, 4),
    ("S5 quattro", 2021, 3.0, 6), ("SQ5", 2021, 3.0, 6),
    ("Q3 quattro", 2021, 2.0, 4), ("A4 quattro", 2021, 3.0, 6),
])
def test_audi_rule_does_not_carry_back_or_spread_to_other_engines(model, year, displacement, cylinders):
    assert reviewed_hybrid_rules("Audi", model, year, displacement, cylinders) == []


@pytest.mark.parametrize("model,year", [("SQ8", 2021), ("SQ7", 2025)])
def test_uncertain_audi_powertrain_is_distinct_from_confirmed_hybrid(model, year):
    assert reviewed_hybrid_rules("Audi", model, year, 4.0, 8) == []
    assert reviewed_uncertain_rules("Audi", model, year, 4.0, 8) == ["audi_sq7_sq8_4l_technology_uncertain"]
    config = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8"))
    record = json.loads((FIXTURES / "44187.json").read_text(encoding="utf-8"))
    record.update(make="Audi", model=model, year=str(year), displ="4.0", cylinders="8")
    row, reasons, _ = normalize_record(record, {"sha256": "b" * 64, "url": "offline-synthetic-scope-case"}, config)
    assert "reviewed_powertrain_uncertain:audi_sq7_sq8_4l_technology_uncertain" in reasons
    specifications = {field: row[field] for field in (*MIDTERM_FEATURES, "model_name", "engine_description")}
    with pytest.raises(ValueError, match="source technology labels need clarification"):
        validate_specifications(specifications, config, True)


def test_uncertain_rule_is_not_a_blanket_manufacturer_exclusion():
    assert reviewed_uncertain_rules("Audi", "SQ5", 2021, 3.0, 6) == []
    assert reviewed_uncertain_rules("Audi", "SQ7", 2019, 4.0, 8) == []
    assert reviewed_uncertain_rules("Audi", "SQ7", 2025, 3.0, 6) == []


@pytest.mark.parametrize("field", ["model_name", "displacement_l", "cylinders"])
def test_reviewed_identity_cannot_bypass_scope_check_with_missing_input(field):
    config = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8"))
    values = {"manufacturer": "Audi", "model_name": "A4 quattro", "model_year": 2021,
              "displacement_l": 2.0, "cylinders": 4, "vehicle_class": "Compact Cars",
              "transmission": "Automatic (S7)", "drivetrain": "All-Wheel Drive"}
    values[field] = "" if field == "model_name" else None
    with pytest.raises(ValueError, match="powertrain"):
        validate_specifications(values, config, True)


def test_cleaning_quarantines_unresolved_engine_condition():
    config = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8"))
    record = json.loads((FIXTURES / "44187.json").read_text(encoding="utf-8"))
    record.update(make="Audi", model="A4 quattro", year="2021", displ="", cylinders="4")
    _, reasons, _ = normalize_record(record, {"sha256": "c" * 64, "url": "offline-synthetic-scope-case"}, config)
    assert "reviewed_powertrain_uncertain:audi_2021_2022_a4_a5_q5_2l_mild_hybrid:missing_engine_condition" in reasons
