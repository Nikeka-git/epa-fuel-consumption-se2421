"""Final release verification; no collection, cleaning, training, or Git mutation.

Run only after all full-benchmark artifacts are finished and staged for release:
  python scripts/verify_release.py --slides <main.pptx> --slides-receipt <validation.json>
Optional: --pytest-json <recorded test evidence.json> OR --pytest-xml <actual junit.xml>,
--slides-visual-review <actual review.json>, --dry-run.
The only repository write is evidence/PACKAGE_VERIFICATION.json after every check
passes, unless --dry-run is given. Missing evidence fails; counts are never guessed.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

from fuel_consumption.api import parse_menu, validate_vehicle
from fuel_consumption.split import load_frozen_split, read_dataset
from fuel_consumption.utils import write_json


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
RUNS = ("midterm_v1", "endterm_v1", "final_v1")
SOURCE_REVIEW = {
    "guide_pdf": "Previously extracted text and all three pages visually inspected; not repeated by this preflight.",
    "rubric_docx": "Previously inspected OOXML text and all four tables; no comments or tracked changes. Visual rendering unavailable because bundled LibreOffice is absent.",
    "lecture_pdfs": "Previously extracted text and key regression pages visually inspected; not repeated by this preflight.",
}
SECRET = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{35,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def confined(root, relative):
    value = Path(relative)
    require(not value.is_absolute() and ".." not in value.parts, "Artifact path must be relative and confined")
    path = (root / value).resolve()
    path.relative_to(root.resolve())
    return path


def module_from(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def verify_raw(root, snapshot_name):
    directory = root / "data/raw" / snapshot_name
    require(not (directory / ".collector.lock").exists(), "Collector must be stopped before final verification")
    metadata = read(directory / "snapshot.json")
    require(metadata.get("snapshot_id") == snapshot_name and metadata.get("status") in {"partial", "complete"}, "Snapshot is not finished for release")
    require(metadata.get("finished_at_utc"), "Final collection completion time is absent")
    config = read(root / "configs/project.json")
    require(read(directory / "config.json") == config, "Collection config differs from the frozen project config")
    require(metadata.get("config_sha256") == sha(root / "configs/project.json"), "Original collection configuration checksum differs")
    require(metadata["requested_scope"]["years"] == list(range(2015, 2026)) and metadata["requested_scope"].get("makes") is None, "Main source scope must include all eleven years and available makes")
    cache = read(directory / "cache_index.json")["entries"]
    inventory = read(directory / "inventory.json")["records"]
    manifest_body = (directory / "requests.jsonl").read_bytes()
    manifest = [json.loads(line) for line in manifest_body.splitlines() if line.strip()]
    successful = {row["request_id"]: row for row in manifest if row.get("valid") is True and row.get("status") == 200}
    require(len({row["request_id"] for row in manifest}) == len(manifest), "Duplicate request IDs in source manifest")
    # Every recorded response body, including failures, is checked by its recorded
    # path. This never discovers evidence by globbing uncommitted response files.
    recorded_bodies = 0
    for entry in manifest:
        if entry.get("file"):
            path = confined(directory, entry["file"])
            require(sha(path) == entry.get("sha256"), "Recorded response checksum differs: " + entry["file"])
            recorded_bodies += 1
    vehicles = {}
    for url, entry in cache.items():
        matching = successful.get(entry.get("request_id"))
        require(matching is not None, "Cached request has no successful manifest entry")
        require(all(matching.get(key) == entry.get(key) for key in ("url", "file", "sha256", "kind", "expected")), "Cache/manifest provenance differs")
        require(entry.get("url") == url and entry.get("valid") is True and entry.get("status") == 200, "Cached entry is not a validated HTTP response")
        payload = read(confined(directory, entry["file"]))
        if entry["kind"] == "menu":
            parse_menu(payload)
        else:
            require(entry["kind"] == "vehicle", "Unknown cached resource kind")
            validate_vehicle(payload, **entry.get("expected", {}))
            identifier = str(payload["id"])
            require(identifier not in vehicles, "Duplicate individual vehicle cache identifier")
            vehicles[identifier] = entry
            row = inventory.get(identifier)
            require(isinstance(row, dict) and row.get("status") == "fetched", "Successful vehicle has no fetched inventory row")
            identities = {(item.get("model_year"), item.get("manufacturer"), item.get("model_name")) for item in row.get("provenance", [])}
            require(identities == {(int(payload["year"]), payload["make"], payload["model"])}, "Conflicting vehicle/menu identity")
    require({identifier for identifier, row in inventory.items() if row.get("status") == "fetched"} == set(vehicles), "Fetched inventory/cache IDs differ")
    require(len(vehicles) == metadata["fetched_vehicle_records"], "Snapshot fetched count differs")
    require(len(manifest) == metadata["request_attempts_total"], "Snapshot durable attempt count differs")
    require(not list((directory / "pending_requests").glob("*.json")), "Unrecovered source response journals remain")
    source_release_paths = {path.relative_to(root).as_posix() for path in [directory / entry["file"] for entry in manifest if entry.get("file")]}
    source_release_paths.update((directory / name).relative_to(root).as_posix() for name in ("snapshot.json", "config.json", "requests.jsonl", "cache_index.json", "inventory.json", "inventory.csv", "collection_state.json"))
    return {"snapshot_id": snapshot_name, "status": metadata["status"], "full_catalogue_complete": metadata.get("full_catalogue_complete", False),
            "verified_response_bodies": recorded_bodies, "successful_cache_entries": len(cache), "raw_vehicle_records": len(vehicles),
            "durable_recorded_attempts": len(manifest), "failed_attempts": len(manifest) - len(successful),
            "unfinished_historical_invocations": sum(not row.get("finished_at_utc") for row in metadata["invocations"]),
            "collection_finished_at_utc": metadata["finished_at_utc"], "snapshot_sha256": sha(directory / "snapshot.json")}, vehicles, source_release_paths


def verify_notebook(path):
    notebook = read(path)
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    require(len(cells) == 12, "Main notebook must have twelve executed code cells")
    counts = [cell.get("execution_count") for cell in cells]
    require(all(isinstance(count, int) and count > 0 for count in counts) and counts == sorted(set(counts)), "Notebook code cells are not all executed in order")
    errors = [output for cell in cells for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    require(not errors, "Notebook contains execution errors")
    source = "\n".join("".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "") for cell in cells)
    require("ALLOW_SMALL = False" in source and "models/midterm_v1" in source and "EXISTING_RUN = None" not in source, "Notebook must read the full saved Midterm run")
    plots = sum("image/png" in output.get("data", {}) for cell in cells for output in cell.get("outputs", []))
    require(plots >= 3, "Notebook has fewer than three visible figure outputs")
    return {"sha256": sha(path), "executed_code_cells": len(cells), "execution_counts": counts, "execution_errors": 0,
            "embedded_png_outputs": plots, "visual_review": "Not performed by this read-only preflight; verify separately."}


def verify_slides(pptx, receipt_path, visual_path=None):
    pptx_digest = sha(pptx)
    with zipfile.ZipFile(pptx) as archive:
        require(archive.testzip() is None, "PPTX archive CRC check failed")
        names = archive.namelist()
        require(len(names) == len(set(names)), "PPTX contains duplicate archive members")
        slides = sorted(name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name))
        notes = [name for name in names if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", name)]
        require(len(slides) == 8 and len(notes) == 8, "Main PPTX must contain eight slides and eight notes parts")
        require({"[Content_Types].xml", "ppt/presentation.xml", "_rels/.rels"}.issubset(names), "PPTX required package parts are missing")
        for name in names:
            if name.endswith((".xml", ".rels")):
                ET.fromstring(archive.read(name))
    receipt = read(receipt_path)
    require(receipt.get("finalSha256") == pptx_digest and receipt.get("byteCount") == pptx.stat().st_size, "Presentation validation receipt refers to different bytes")
    integrity = receipt["packageIntegrity"]
    require(integrity.get("status") == "pass" and integrity.get("slide_count") == 8 and integrity.get("finding_count") == 0, "Presentation package validation failed")
    layout, font = receipt["presentationLayout"], receipt["fontSelection"]
    require(layout.get("exitCode") == 0 and layout.get("findingCount") == 0, "Presentation layout validation failed")
    require(font.get("performed") is True and font.get("passed") is True and font.get("observed_font_families"), "Presentation font policy has not passed")
    require(set(font["observed_font_families"]).issubset(set(font["approvedFamilies"])), "Presentation uses fonts outside its approved policy")
    visual = {"checked_by_preflight": False, "status": "Separate visual review not supplied"}
    if visual_path:
        visual = read(visual_path)
        require(visual.get("pptx_sha256") == pptx_digest and visual.get("reviewed_slides") == 8 and not visual.get("findings"), "Full slide visual review does not match the release presentation")
    return {"sha256": pptx_digest, "slide_count": len(slides), "speaker_note_parts": len(notes), "validation_receipt_sha256": sha(receipt_path),
            "package_status": integrity["status"], "font_policy_passed": font["passed"], "font_families": font["observed_font_families"],
            "native_font_rendering_verified": font.get("native_font_rendering_verified", False), "visual_review": visual,
            "claim_boundary": "Archive, OOXML and saved portable validation evidence only; no native PowerPoint opening or native font rendering is asserted."}


def verify_git_files(root):
    completed = subprocess.run(["git", "ls-files", "--cached", "-z"], cwd=root, check=True, capture_output=True)
    names = sorted({name.decode("utf-8") for name in completed.stdout.split(b"\0") if name})
    tracked = set(names)
    require(names, "No staged/tracked release files")
    total, links, largest = 0, 0, 0
    for name in names:
        relative = Path(name)
        require(not {"work", "__pycache__", ".venv", ".pytest_cache", ".ipynb_checkpoints"}.intersection(relative.parts), "Private/cache directory tracked: " + name)
        require(not (relative.name.endswith((".tmp", ".lock", ".pyc")) or relative.name.startswith(".collector.lock")), "Temporary/lock file tracked: " + name)
        path = confined(root, name)
        size = path.stat().st_size
        require(size < 50 * 1024 * 1024, "Tracked file exceeds the <50 MiB release limit: " + name)
        total += size; largest = max(largest, size)
        body = path.read_bytes()
        require(not SECRET.search(body), "Credential-like content detected in release file: " + name)
        if path.suffix == ".py":
            ast.parse(body.decode("utf-8-sig"))
        elif path.suffix == ".json":
            json.loads(body)
        elif path.suffix == ".md":
            content = body.decode("utf-8-sig")
            require(content.count("```") % 2 == 0, "Unbalanced Markdown fences: " + name)
            content = re.sub(r"```.*?```", "", content, flags=re.S)
            for destination in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
                destination = destination.strip()
                destination = destination[1:destination.index(">") ] if destination.startswith("<") and ">" in destination else re.split(r'\s+["\']', destination, maxsplit=1)[0]
                if re.match(r"^[a-zA-Z][a-zA-Z+.-]*:", destination) or destination.startswith("#"):
                    continue
                local = unquote(destination.split("#", 1)[0].split("?", 1)[0])
                if local:
                    require((path.parent / local).exists(), f"Broken local Markdown link: {name} -> {local}")
                    resolved = (path.parent / local).resolve()
                    if resolved.is_relative_to(root):
                        destination_name = resolved.relative_to(root).as_posix()
                        published = destination_name in tracked or (resolved.is_dir() and any(item.startswith(destination_name.rstrip("/") + "/") for item in tracked))
                        require(published, "Local Markdown destination is absent from the staged release: " + name)
                    links += 1
    unchanged = subprocess.run(["git", "diff", "--name-only", "-z"], cwd=root, check=True, capture_output=True)
    require(not unchanged.stdout, "Tracked working files differ from the staged release; stage final changes before verification")
    return {"tracked_files": len(names), "total_bytes": total, "largest_file_bytes": largest, "local_markdown_links_checked": links,
            "secret_scan": "No obvious credential-like patterns found; this is not a comprehensive secret audit."}, tracked


def verify_pytest_json(root, path):
    """Verify recorded completed-suite evidence without rerunning that suite."""
    result = read(path)
    require(result.get("status") == "passed" and result.get("exit_code") == 0, "Recorded pytest run did not pass")
    require(type(result.get("tests")) is int and result["tests"] > 0, "Recorded pytest test count is absent")
    duration = result.get("duration_seconds")
    require(isinstance(duration, (int, float)) and np.isfinite(duration) and duration > 0, "Recorded pytest duration is absent")
    require(isinstance(result.get("command"), str) and "pytest" in result["command"], "Exact completed pytest command is absent")
    summary = result.get("stdout_summary", "")
    passed = re.search(r"\b(\d+) passed\b", summary)
    elapsed = re.search(r"\bin (\d+(?:\.\d+)?)s\b", summary)
    require(passed and elapsed and int(passed.group(1)) == result["tests"] and np.isclose(float(elapsed.group(1)), duration, rtol=0, atol=0.005), "Recorded test count/duration differs from the actual stdout summary")
    require(not re.search(r"\b\d+ (?:failed|error|errors)\b", summary), "Recorded pytest stdout contains failures/errors")
    mapping = result.get("source_sha256")
    require(isinstance(mapping, dict) and mapping, "Recorded test-source SHA map is absent")
    required = {file.relative_to(root).as_posix() for directory in (root / "src", root / "tests") for file in directory.rglob("*.py")}
    required.update(file.relative_to(root).as_posix() for file in (root / "configs").rglob("*.json"))
    required.update(file.relative_to(root).as_posix() for file in (root / "tests/fixtures").rglob("*") if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc")
    if (root / "app/streamlit_app.py").is_file():
        required.add("app/streamlit_app.py")
    require(required.issubset(mapping), "Test-source SHA map omits current source/config/test/fixture/app files")
    for name, digest in mapping.items():
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "Recorded test-source digest is invalid")
        require(sha(confined(root, name)) == digest, "Source/config/test fixture changed after the completed suite: " + name)
    return {"status": "passed", "evidence_format": "recorded_exec_json", "evidence_sha256": sha(path), "tests": result["tests"],
            "duration_seconds": duration, "command": result["command"], "stdout_summary": summary,
            "verified_source_files": len(mapping), "source_commit": result.get("source_commit"), "recorded_at_utc": result.get("recorded_at_utc"),
            "evidence_origin": result.get("evidence_origin"), "verification_basis": "Recorded completed execution and current source hashes; the suite was not rerun by this preflight."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--snapshot", default="benchmark_2015_2025_20261009")
    parser.add_argument("--slides", type=Path, required=True)
    parser.add_argument("--slides-receipt", type=Path, required=True)
    parser.add_argument("--slides-visual-review", type=Path)
    pytest_evidence = parser.add_mutually_exclusive_group()
    pytest_evidence.add_argument("--pytest-xml", type=Path)
    pytest_evidence.add_argument("--pytest-json", type=Path, help="Actual completed exec evidence with source_sha256; no suite rerun")
    parser.add_argument("--dry-run", action="store_true", help="Verify without writing PACKAGE_VERIFICATION.json")
    args = parser.parse_args()
    root = args.root.resolve()
    report_guard = module_from(root / "scripts/build_project_report.py", "release_report_guards")
    raw, vehicles, raw_release_paths = verify_raw(root, args.snapshot)
    dataset, config, splits = root / "data/processed/vehicles.parquet", root / "configs/project.json", root / "data/splits"
    frame = read_dataset(dataset)
    require(len(frame) >= 3000, "Full release requires at least 3,000 cleaned configurations")
    require(set(frame.snapshot_id) == {args.snapshot}, "Processed snapshot identity differs")
    manifest, split = load_frozen_split(dataset, config, splits)
    dataset_hash, split_hash = sha(dataset), sha(splits / "split_manifest.csv")
    cleaning = read(root / "data/interim/cleaned_summary.json")
    require(cleaning["processed_parquet_sha256"] == dataset_hash and cleaning["cleaned_rows"] == len(frame), "Cleaning summary differs from main dataset")
    require(cleaning["processed_csv_sha256"] == sha(root / "data/processed/vehicles.csv"), "Processed CSV checksum differs")
    schema = read(root / "data/processed/schema.json")
    require(schema["snapshot_id"] == args.snapshot and schema["artifact_hashes"] == {"vehicles.csv": cleaning["processed_csv_sha256"], "vehicles.parquet": dataset_hash}, "Processed schema or artifact fingerprints differ")
    require(cleaning["cleaning_module_sha256"] == sha(root / "src/fuel_consumption/clean.py"), "Cleaner changed after the benchmark was prepared")
    powertrain = cleaning["reviewed_powertrain_rules"]
    require(powertrain["rules_sha256"] == sha(confined(root, powertrain["rules_relative_path"])) and powertrain["module_sha256"] == sha(root / "src/fuel_consumption/powertrain.py"), "Reviewed powertrain rules/module changed after cleaning")
    require(cleaning["unique_vehicle_ids"] == len(frame) and cleaning["desired_rows_met"] is True, "Desired distinct-count evidence differs")
    require(cleaning["raw_vehicle_files"] == raw["raw_vehicle_records"], "Raw/cleaning record counts differ")
    require(set(frame.vehicle_id).issubset(vehicles), "Processed IDs missing from verified raw cache")
    require(np.isfinite(frame.target_l100km).all() and frame.target_l100km.gt(0).all(), "Processed targets are not positive finite")
    require(frame.apply(lambda row: row.raw_sha256 == vehicles[row.vehicle_id]["sha256"], axis=1).all(), "Processed raw SHA provenance differs")
    runs = [report_guard.checked_run(root, name, dataset_hash, split_hash, config_hash=sha(config),
                                    split_metadata_hash=sha(splits / "split_metadata.json"), frame=frame, manifest=manifest) for name in RUNS]
    require({metadata["stage"] for metadata, _, _ in runs} == {"midterm", "endterm", "final"}, "Three full stages are required")
    for metadata, _, _ in runs:
        for name, digest in metadata["code_sha256"].items():
            require(sha(confined(root / "src/fuel_consumption", name)) == digest, "Training source changed after the saved main run: " + metadata["run_id"] + "/" + name)
        if metadata["stage"] != "midterm":
            stage_hash, _ = report_guard.checked_stage_config_hashes(root, root / "models" / metadata["run_id"], metadata)
            require(stage_hash == sha(root / "configs" / (metadata["stage"] + ".json")), "Computational stage configuration changed after the run")
    final = next(item for item in runs if item[0]["stage"] == "final")
    require(final[0]["interface_schema_sha256"] == sha(root / "models/final_v1/interface_schema.json"), "Final inference schema checksum differs")
    report_guard.checked_paired(final[1], pd.read_csv(final[1] / "predictions.csv", dtype={"vehicle_id": str}), split["n_test"])
    endterm = next(item for item in runs if item[0]["stage"] == "endterm")
    report_guard.checked_segments(root, endterm[0], manifest)
    report_directory = root / "reports/project_v1"
    report = read(report_directory / "experiment_summary.json")
    require(report["dataset_sha256"] == dataset_hash and report["split_sha256"] == split_hash and report["development_report"] is False, "Combined report benchmark/stage differs")
    require(report["model_count"] == 11 and report["stage_count"] == 3, "Combined report must describe eleven models across three stages")
    require(report["runs"] == [metadata for metadata, _, _ in runs], "Combined report run metadata differs")
    require(report["report_builder_sha256"] == sha(root / "scripts/build_project_report.py"), "Combined report builder changed after its recorded generation")
    for name, digest in report["artifacts_sha256"].items():
        require(sha(confined(report_directory, name)) == digest, "Combined report artifact checksum differs: " + name)
    for metadata, tables, _ in runs:
        require(report["input_metrics_sha256"][metadata["run_id"]] == sha(tables / "metrics.csv"), "Combined report input metric hash differs")
    notebook = verify_notebook(root / "notebooks/01_midterm.ipynb")
    slides = verify_slides(args.slides.resolve(), args.slides_receipt.resolve(), args.slides_visual_review)
    slide_summary = read(root / "reports/slides/slide_summary.json")
    require(slide_summary["dataset"]["dataset_sha256"] == dataset_hash and slide_summary["dataset"]["development_small_dataset"] is False, "Slides source summary differs from main benchmark")
    for source in slide_summary["source_files"]:
        # The summary preserves its original Windows collection paths. Resolve
        # the exact repository suffix locally, so a clone never reads that host.
        original = source["path"].replace("\\", "/")
        marker = "/outputs/fuel_consumption_project/"
        require(original.count(marker) == 1, "Unexpected original slide-source path")
        relative = original.split(marker, 1)[1]
        require(sha(confined(root, relative)) == source["sha256"], "Slides numerical-source hash differs")
    for reference in read(root / "references/source_manifest.json"):
        require(sha(confined(root / "references", reference["file"])) == reference["sha256"], "Provided source-reference checksum differs")
    git, tracked = verify_git_files(root)
    require(args.slides.resolve().is_relative_to(root), "Release PPTX must be inside the repository deliverables")
    required_tracked = {"data/processed/vehicles.parquet", "data/processed/vehicles.csv", "data/processed/schema.json", "data/interim/cleaned_summary.json", "data/splits/split_manifest.csv", "data/splits/split_metadata.json", "reports/project_v1/experiment_summary.json", "reports/slides/slide_summary.json", "notebooks/01_midterm.ipynb", args.slides.resolve().relative_to(root).as_posix()}
    required_tracked.update((report_directory / name).relative_to(root).as_posix() for name in report["artifacts_sha256"])
    for receipt_path in (args.slides_receipt, args.slides_visual_review, args.pytest_json):
        if receipt_path and receipt_path.resolve().is_relative_to(root):
            required_tracked.add(receipt_path.resolve().relative_to(root).as_posix())
    required_tracked.update(f"models/{name}/run_metadata.json" for name in RUNS)
    required_tracked.update(raw_release_paths)
    required_tracked.update((splits / name).relative_to(root).as_posix() for name in split["artifact_sha256"])
    for metadata, tables, _ in runs:
        model_directory = root / "models" / metadata["run_id"]
        required_tracked.update((model_directory / (model + ".joblib")).relative_to(root).as_posix() for model in metadata["model_artifact_sha256"])
        required_tracked.update((model_directory / name).relative_to(root).as_posix() for name in ("config.json", "cv_selection.json"))
        required_tracked.update((tables / name).relative_to(root).as_posix() for name in ("metrics.csv", "predictions.csv", "fold_metrics.csv", "subgroup_errors.csv", "large_errors.csv"))
        if metadata["stage"] != "midterm":
            required_tracked.add((model_directory / "stage_config.json").relative_to(root).as_posix())
        if metadata["stage"] == "endterm":
            required_tracked.update((model_directory / "segments" / name).relative_to(root).as_posix() for name in metadata["segments"]["artifacts_sha256"])
            required_tracked.add((model_directory / "segments/segment_metadata.json").relative_to(root).as_posix())
        if metadata["stage"] == "final":
            required_tracked.add((model_directory / "interface_schema.json").relative_to(root).as_posix())
    require(required_tracked.issubset(tracked), "Main release artifacts must be staged/tracked before release verification")
    tests = {"status": "Not supplied; no pytest execution or pass count is asserted by this preflight"}
    if args.pytest_json:
        tests = verify_pytest_json(root, args.pytest_json.resolve())
    elif args.pytest_xml:
        document = ET.parse(args.pytest_xml).getroot()
        suites = [document] if document.tag == "testsuite" else list(document.iter("testsuite"))
        failures = sum(int(suite.get("failures", 0)) + int(suite.get("errors", 0)) for suite in suites)
        require(suites and failures == 0, "Supplied pytest JUnit evidence contains failures/errors")
        tests = {"status": "passed", "junit_sha256": sha(args.pytest_xml), "tests": sum(int(suite.get("tests", 0)) for suite in suites),
                 "skipped": sum(int(suite.get("skipped", 0)) for suite in suites), "failures_and_errors": failures}
    evidence = {"status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(), "release_stage": "full benchmark and completed three-stage ML project",
                "raw": raw, "dataset": {"cleaned_rows": len(frame), "minimum_3000_met": True, "sha256": dataset_hash,
                                          "unresolved_duplicate_candidate_groups": cleaning.get("unresolved_candidate_groups", 0)},
                "split": split, "runs": [{"run_id": metadata["run_id"], "stage": metadata["stage"], "selected_model": metadata["selected_model"],
                                           "model_count": len(table), "dataset_sha256": metadata["dataset_sha256"], "split_sha256": metadata["split_sha256"]}
                                          for metadata, _, table in runs], "combined_report": {"models": 11, "stages": 3, "summary_sha256": sha(report_directory / "experiment_summary.json")},
                "notebook": notebook, "presentation": slides, "git_release": git, "offline_tests": tests, "source_review": SOURCE_REVIEW,
                "remaining_claim_limits": ["Source DOCX native/visual rendering was unavailable.", "Portable slide checks do not prove native PowerPoint rendering.",
                                           "Verification covers saved technical artifacts and the recorded source hashes."],
                "preflight_script_sha256": sha(Path(__file__))}
    if not args.dry_run:
        destination = root / "evidence/PACKAGE_VERIFICATION.json"
        write_json(destination, evidence)
    print(json.dumps({"status": evidence["status"], "raw_records": raw["raw_vehicle_records"], "cleaned_rows": len(frame), "models": 11,
                      "executed_code_cells": notebook["executed_code_cells"], "slides": slides["slide_count"], "tracked_files": git["tracked_files"], "evidence_written": not args.dry_run}))


if __name__ == "__main__":
    main()
