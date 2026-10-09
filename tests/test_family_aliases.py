"""Source-identity graph invariants, without targets or experimental splits."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fuel_consumption.utils import project_root

spec = spec_from_file_location("family_alias_builder", project_root() / "scripts/build_family_aliases.py")
builder = module_from_spec(spec)
spec.loader.exec_module(builder)


def row(identifier, model, base, make="Acme"):
    return {"vehicle_id": str(identifier), "manufacturer": make, "model_name": model,
            "base_model": base or "", "model_year": 2020,
            "normalized_manufacturer": builder.normalize(make),
            "normalized_model_name": builder.normalize(model),
            "normalized_base_model": builder.normalize(base), "raw_sha256": "a" * 64,
            "source_file": "synthetic fixture", "source_url": "synthetic fixture"}


def test_connected_literal_aliases_are_transitive_and_cover_other_full_variants():
    data = [row(1, "Bridge one", "C"), row(2, "Bridge one", "B"),
            row(3, "Bridge two", "B"), row(4, "Bridge two", "A"), row(5, "Other variant", "C")]
    components, aliases, edges, evidence = builder.generate_candidates(data)
    assert len(components) == 1 and len(edges) == 2
    assert components[0]["canonical_base_model"] == "a"
    assert components[0]["source_base_models"] == "a | b | c"
    assert {r["model_name"] for r in aliases} == {"Bridge one", "Bridge two", "Other variant"}
    assert all(r["canonical_base_model"] == "a" and r["decision_status"] == "pending" for r in aliases)
    assert len(evidence) == 5


def test_similar_model_name_prefix_does_not_create_an_alias():
    data = [row(1, "Alpha", "A"), row(2, "Alpha", "B"), row(3, "Alpha Sport", "C")]
    components, aliases, _, _ = builder.generate_candidates(data)
    assert len(components) == 1
    assert components[0]["source_base_models"] == "a | b"
    assert {r["model_name"] for r in aliases} == {"Alpha"}


def test_identical_model_names_in_different_manufacturers_stay_separate():
    data = [row(1, "Alpha", "A", "Acme"), row(2, "Alpha", "B", "Acme"),
            row(3, "Alpha", "B", "Other"), row(4, "Alpha", "C", "Other")]
    components, _, _, _ = builder.generate_candidates(data)
    assert len(components) == 2
    assert {r["canonical_base_model"] for r in components} == {"a", "b"}
    assert len({r["component_id"] for r in components}) == 2


def test_missing_base_can_follow_exact_normalized_full_name_evidence():
    data = [row(1, "Alpha Sport", "A"), row(2, "Alpha Sport", "B"), row(3, " ALPHA   SPORT ", None)]
    components, aliases, _, evidence = builder.generate_candidates(data)
    assert components[0]["n_rows"] == 3
    assert len(aliases) == 1 and aliases[0]["canonical_base_model"] == "a"
    assert {r["vehicle_id"] for r in evidence} == {"1", "2", "3"}


def test_running_checkpoint_cannot_become_final_aliases(tmp_path):
    snapshot = tmp_path / "raw"
    (snapshot / "vehicles").mkdir(parents=True)
    (snapshot / "snapshot.json").write_text(json.dumps({"snapshot_id": "test", "status": "running", "finished_at_utc": None}))
    for identifier, base in [(1, "A"), (2, "B")]:
        (snapshot / "vehicles" / f"{identifier}.json").write_text(json.dumps({"id": str(identifier), "make": "Acme", "model": "Alpha", "baseModel": base, "year": "2020"}))
    ids = tmp_path / "ids.csv"
    ids.write_text("vehicle_id,scope_eligible\n1,True\n2,True\n")
    output = tmp_path / "review"
    args = SimpleNamespace(snapshot=snapshot, ids=ids, output_dir=output, final=True, audit_summary=None, decisions=None)
    with pytest.raises(ValueError, match="collection to stop"):
        builder.build(args)
    assert not output.exists()
