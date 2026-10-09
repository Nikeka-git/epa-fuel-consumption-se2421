"""Apply reviewed manufacturer technology evidence without reading the target."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

from .utils import project_root

RULES_PATH = project_root() / "configs" / "powertrain_exclusions.json"


@lru_cache(maxsize=1)
def reviewed_rules():
    body = RULES_PATH.read_bytes()
    document = json.loads(body)
    rules, identifiers = [], set()
    required = {"rule_id", "manufacturer", "model_year_min", "model_year_max", "model_regex",
                 "technology", "source_ids", "rationale"}
    for rule in document["rules"]:
        if not required.issubset(rule) or set(rule) - required - {"displacement_l", "cylinders"} or rule["rule_id"] in identifiers or rule["technology"] not in {"mild_hybrid", "powertrain_uncertain"}:
            raise ValueError("Invalid reviewed powertrain exclusion rule")
        if not isinstance(rule["manufacturer"], str) or not rule["manufacturer"].strip():
            raise ValueError("Reviewed rule must specify a manufacturer")
        lower, upper = rule["model_year_min"], rule["model_year_max"]
        if type(lower) is not int or type(upper) is not int or not 2015 <= lower <= upper <= 2025:
            raise ValueError("Reviewed rule requires a bounded model-year interval")
        if not rule["source_ids"] or not set(rule["source_ids"]).issubset(document["sources"]):
            raise ValueError("Reviewed rule must cite available source evidence")
        for field in ("displacement_l", "cylinders"):
            if field in rule and (not isinstance(rule[field], (int, float)) or isinstance(rule[field], bool) or not math.isfinite(rule[field]) or rule[field] <= 0):
                raise ValueError("Reviewed engine condition must be positive numeric")
        identifiers.add(rule["rule_id"])
        rules.append((rule, re.compile(rule["model_regex"], re.IGNORECASE)))
    return document, tuple(rules), hashlib.sha256(body).hexdigest()


def _matches(manufacturer, model_name, model_year, displacement_l=None, cylinders=None, identity_only=False):
    """Use identity and optional technical engine fields; never target values."""
    if not isinstance(manufacturer, str) or not isinstance(model_name, str) or model_year is None:
        return []
    _, rules, _ = reviewed_rules()
    engine = {"displacement_l": displacement_l, "cylinders": cylinders}
    return [rule for rule, pattern in rules
            if manufacturer.strip().casefold() == rule["manufacturer"].casefold()
            and rule["model_year_min"] <= model_year <= rule["model_year_max"]
            and pattern.search(model_name)
            and (identity_only or all(field not in rule or (engine[field] is not None and abs(float(engine[field]) - rule[field]) < 1e-6)
                    for field in engine))]


def requires_model_designation(manufacturer, model_year) -> bool:
    _, rules, _ = reviewed_rules()
    return any(isinstance(manufacturer, str) and manufacturer.strip().casefold() == rule["manufacturer"].casefold()
               and rule["model_year_min"] <= model_year <= rule["model_year_max"] for rule, _ in rules)


def reviewed_missing_engine_rules(manufacturer, model_name, model_year, displacement_l=None, cylinders=None) -> list[str]:
    engine = {"displacement_l": displacement_l, "cylinders": cylinders}
    return [rule["rule_id"] for rule in _matches(manufacturer, model_name, model_year, identity_only=True)
            if any(field in rule and (engine[field] is None or not math.isfinite(float(engine[field]))) for field in engine)
            and all(field not in rule or engine[field] is None or not math.isfinite(float(engine[field]))
                    or abs(float(engine[field]) - rule[field]) < 1e-6 for field in engine)]


def reviewed_hybrid_rules(manufacturer, model_name, model_year, displacement_l=None, cylinders=None) -> list[str]:
    return [rule["rule_id"] for rule in _matches(manufacturer, model_name, model_year, displacement_l, cylinders)
            if rule["technology"] == "mild_hybrid"]


def reviewed_uncertain_rules(manufacturer, model_name, model_year, displacement_l=None, cylinders=None) -> list[str]:
    return [rule["rule_id"] for rule in _matches(manufacturer, model_name, model_year, displacement_l, cylinders)
            if rule["technology"] == "powertrain_uncertain"]


def ruleset_evidence() -> dict:
    document, _, digest = reviewed_rules()
    if hashlib.sha256(RULES_PATH.read_bytes()).hexdigest() != digest:
        raise ValueError("Powertrain rules changed during dataset preparation")
    return {"ruleset_version": document["ruleset_version"], "rules_relative_path": "configs/powertrain_exclusions.json",
            "rules_sha256": digest, "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rule_ids": [r["rule_id"] for r in document["rules"]], "target_used_for_rules": False,
            "sources": document["sources"], "limitations": document["limitations"]}
