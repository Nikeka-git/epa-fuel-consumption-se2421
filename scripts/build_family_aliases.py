"""Generate target-free literal-source family alias candidates for explicit review.

This script never edits raw/config/cleaned data or a saved split. It connects
source baseModel names only when the same normalized manufacturer and full model
name appear under multiple baseModels. A component includes every model variant
under its observed source bases, with a deterministic existing-base canonical.
No prefix, fuzzy-name, target, error or split-outcome heuristic is used.

Default outputs are provisional candidates. --final requires a completed source
snapshot, a final independent audit and explicit component review decisions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


def normalize(value) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold() if value is not None else ""


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        rows = pd.read_parquet(path, columns=["vehicle_id"])
    else:
        rows = pd.read_csv(path, dtype={"vehicle_id": str}, usecols=lambda col: col in {"vehicle_id", "scope_eligible"})
        if "scope_eligible" in rows:
            if not set(rows["scope_eligible"].astype(str).str.casefold()).issubset({"true", "false"}):
                raise ValueError("scope_eligible must explicitly contain true/false")
            rows = rows[rows["scope_eligible"].astype(str).str.casefold().eq("true")]
    ids = rows["vehicle_id"].astype(str).tolist()
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[1-9]\d*", value) for value in ids):
        raise ValueError("Review IDs must be unique source vehicle IDs")
    return sorted(ids, key=int)


def read_identities(snapshot: Path, ids: list[str]) -> list[dict]:
    rows = []
    for vehicle_id in ids:
        source = snapshot / "vehicles" / f"{vehicle_id}.json"
        record = json.loads(source.read_text(encoding="utf-8"))
        if str(record.get("id")) != vehicle_id:
            raise ValueError(f"Identity mismatch in {source}")
        if "baseModel" in record and "basemodel" in record and record["baseModel"] != record["basemodel"]:
            raise ValueError(f"Conflicting baseModel aliases for {vehicle_id}")
        make, model = record.get("make"), record.get("model")
        base = record.get("baseModel", record.get("basemodel"))
        if not isinstance(make, str) or not isinstance(model, str) or not make.strip() or not model.strip():
            raise ValueError(f"Missing source name for {vehicle_id}")
        if base is not None and not isinstance(base, str):
            raise ValueError(f"Non-string baseModel for {vehicle_id}")
        rows.append({"vehicle_id": vehicle_id, "model_year": int(record["year"]),
                     "manufacturer": make, "model_name": model, "base_model": base or "",
                     "normalized_manufacturer": normalize(make), "normalized_model_name": normalize(model),
                     "normalized_base_model": normalize(base), "raw_sha256": checksum(source),
                     "source_file": str(source.resolve()),
                     "source_url": f"https://www.fueleconomy.gov/ws/rest/vehicle/{vehicle_id}"})
    return rows


def generate_candidates(records: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Return components, aliases, triggering edges and identity evidence only."""
    names, graph = defaultdict(list), defaultdict(lambda: defaultdict(set))
    for row in records:
        names[(row["normalized_manufacturer"], row["normalized_model_name"])].append(row)
        base = row["normalized_base_model"]
        if base:
            graph[row["normalized_manufacturer"]][base]
    triggers = []
    for (make, model), rows in sorted(names.items()):
        bases = sorted({r["normalized_base_model"] for r in rows if r["normalized_base_model"]})
        if len(bases) > 1:
            for base in bases:
                graph[make][base].update(set(bases) - {base})
            triggers.append({"normalized_manufacturer": make, "normalized_model_name": model,
                             "base_models": " | ".join(bases), "vehicle_ids": " | ".join(sorted({r["vehicle_id"] for r in rows}, key=int)),
                             "years": " | ".join(str(y) for y in sorted({r["model_year"] for r in rows})),
                             "edge_rule": "literal_normalized_full_model_name_under_multiple_source_baseModels"})
    components, aliases, evidence = [], [], []
    for make, neighbours in sorted(graph.items()):
        visited = set()
        for start in sorted(neighbours):
            if start in visited:
                continue
            pending, members = [start], set()
            while pending:
                node = pending.pop()
                if node in members:
                    continue
                members.add(node)
                pending.extend(neighbours[node] - members)
            visited.update(members)
            if len(members) < 2:
                continue
            bases = sorted(members)
            component_id = hashlib.sha256(json.dumps([make, bases], separators=(",", ":")).encode()).hexdigest()[:20]
            canonical = bases[0]
            members_rows = [r for r in records if r["normalized_manufacturer"] == make and r["normalized_base_model"] in members]
            member_names = {r["normalized_model_name"] for r in members_rows}
            # A missing-base row with an identical observed full name can follow
            # the reviewed source component, without a prefix-name guess.
            members_rows.extend(r for r in records if r["normalized_manufacturer"] == make and not r["normalized_base_model"] and r["normalized_model_name"] in member_names)
            linked = [t for t in triggers if t["normalized_manufacturer"] == make and set(t["base_models"].split(" | ")).issubset(members)]
            components.append({"component_id": component_id, "manufacturer": members_rows[0]["manufacturer"],
                               "normalized_manufacturer": make, "source_base_models": " | ".join(bases),
                               "canonical_base_model": canonical, "n_rows": len(members_rows),
                               "n_full_model_names": len(member_names), "trigger_full_model_names": " | ".join(t["normalized_model_name"] for t in linked),
                               "decision_status": "pending", "reviewer": "", "rationale": ""})
            for model in sorted(member_names):
                rows = [r for r in members_rows if r["normalized_model_name"] == model]
                source_row = sorted(rows, key=lambda r: int(r["vehicle_id"]))[0]
                aliases.append({"manufacturer": source_row["manufacturer"], "model_name": source_row["model_name"],
                                "canonical_base_model": canonical, "component_id": component_id,
                                "source_base_models": " | ".join(bases), "n_rows": len(rows),
                                "decision_status": "pending", "rule": "connected_literal_source_baseModel_aliases"})
            trigger_names = {t["normalized_model_name"] for t in linked}
            for row in members_rows:
                evidence.append({"component_id": component_id, **row, "triggering_full_model_name": row["normalized_model_name"] in trigger_names,
                                 "canonical_base_model": canonical})
            for row in linked:
                row["component_id"] = component_id
    return components, aliases, triggers, evidence


def build(args) -> dict:
    snapshot, ids_file, output = map(Path, (args.snapshot, args.ids, args.output_dir))
    metadata = json.loads((snapshot / "snapshot.json").read_text(encoding="utf-8"))
    ids = eligible_ids(ids_file)
    rows = read_identities(snapshot, ids)
    components, aliases, triggers, evidence = generate_candidates(rows)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Review output already contains files; choose a new directory")
    if args.final:
        if metadata.get("status") == "running" or not metadata.get("finished_at_utc"):
            raise ValueError("Final aliases require collection to stop before the audit")
        if not args.audit_summary or not args.decisions:
            raise ValueError("Final aliases require --audit-summary and explicit --decisions")
        audit = json.loads(Path(args.audit_summary).read_text(encoding="utf-8"))
        if audit.get("mode") != "final_stopped_snapshot" or audit.get("snapshot_id") != metadata["snapshot_id"] or audit.get("integrity_issue_rows") != 0:
            raise ValueError("Final alias mapping requires the matching final integrity audit")
        captured_snapshot_hash = audit.get("captured_source_sha256", {}).get("snapshot.json")
        if captured_snapshot_hash != checksum(snapshot / "snapshot.json"):
            raise ValueError("Snapshot metadata changed after the supplied final audit")
    decisions = {}
    if args.decisions:
        decision_rows = pd.read_csv(args.decisions, keep_default_na=False)
        required = {"component_id", "decision_status", "reviewer", "rationale"}
        if not required.issubset(decision_rows):
            raise ValueError(f"Decision file needs {sorted(required)}")
        for row in decision_rows.to_dict("records"):
            if row["component_id"] in decisions or row["decision_status"] not in {"approved", "rejected", "pending"}:
                raise ValueError("Review decisions must be unique with approved/rejected/pending status")
            if row["decision_status"] != "pending" and (not str(row["reviewer"]).strip() or not str(row["rationale"]).strip()):
                raise ValueError("Every completed decision needs reviewer and rationale")
            decisions[row["component_id"]] = row
        if set(decisions) != {c["component_id"] for c in components}:
            raise ValueError("Review decision IDs must cover the current source components exactly")
    if args.final and any(row["decision_status"] == "pending" for row in decisions.values()):
        raise ValueError("Unresolved review decisions cannot become a final alias mapping")
    for row in components:
        if row["component_id"] in decisions:
            row.update({key: decisions[row["component_id"]][key] for key in ("decision_status", "reviewer", "rationale")})
    for row in aliases:
        if row["component_id"] in decisions:
            row["decision_status"] = decisions[row["component_id"]]["decision_status"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(components, columns=["component_id", "manufacturer", "normalized_manufacturer", "source_base_models", "canonical_base_model", "n_rows", "n_full_model_names", "trigger_full_model_names", "decision_status", "reviewer", "rationale"]).to_csv(output / "family_components.csv", index=False)
    pd.DataFrame(aliases, columns=["manufacturer", "model_name", "canonical_base_model", "component_id", "source_base_models", "n_rows", "decision_status", "rule"]).to_csv(output / "family_alias_candidates.csv", index=False)
    pd.DataFrame(triggers, columns=["component_id", "normalized_manufacturer", "normalized_model_name", "base_models", "vehicle_ids", "years", "edge_rule"]).to_csv(output / "family_alias_edges.csv", index=False)
    pd.DataFrame(evidence, columns=["component_id", "vehicle_id", "model_year", "manufacturer", "model_name", "base_model", "normalized_manufacturer", "normalized_model_name", "normalized_base_model", "raw_sha256", "source_file", "source_url", "triggering_full_model_name", "canonical_base_model"]).to_csv(output / "family_identity_evidence.csv", index=False)
    pd.DataFrame(components, columns=["component_id", "decision_status", "reviewer", "rationale"]).to_csv(output / "review_decisions_template.csv", index=False)
    if args.final:
        accepted = [r for r in aliases if r["decision_status"] == "approved"]
        pd.DataFrame(accepted, columns=["manufacturer", "model_name", "canonical_base_model"]).to_csv(output / "reviewed_family_aliases.csv", index=False)
    report = {"schema_version": "1.0", "snapshot_id": metadata["snapshot_id"], "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "mode": "final_reviewed_alias_mapping" if args.final else "provisional_candidates_not_applied", "collection_status": metadata["status"],
              "identity_fields_used": ["id", "year", "make", "model", "baseModel"], "target_or_metric_used": False,
              "scope_ids_sha256": checksum(ids_file), "n_eligible_ids": len(ids), "n_components": len(components), "n_triggering_full_names": len(triggers),
              "n_alias_rows": len(aliases), "n_source_evidence_rows": len(evidence), "canonical_rule": "lexicographically_first_normalized_existing_source_baseModel_in_component",
              "raw_or_config_or_split_modified": False, "review_decisions_sha256": checksum(Path(args.decisions)) if args.decisions else None,
              "limitations": ["Literal observed source names support conservative grouping, not vehicle-generation independence.", "The rule does not guess aliases from name prefixes or similar spellings. Additional related families may remain separate.", "A broad source baseModel can merge distinct products. Keeping it together reduces leakage risk but changes the groups being evaluated.", "Regenerate after final collection and independent audit. Checkpoint counts and candidates are provisional."]}
    (output / "family_review_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--ids", required=True, help="Scope audit CSV or cleaned CSV/parquet; only ID/scope-eligibility columns are read")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decisions", help="Explicit component status/reviewer/rationale CSV")
    parser.add_argument("--audit-summary", help="Matching final independent audit_summary.json")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
