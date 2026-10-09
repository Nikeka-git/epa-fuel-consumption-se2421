"""Retrieve catalogue menus and individual vehicle records with durable resume.

The unbounded run enumerates all options first, then fetches each unique ID.
Bounded runs interleave discovery and downloads across years/manufacturers, so
smoke data do not silently mean only the first alphabetic make or first year.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import socket
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .api import (APIError, CorruptCacheError, RawAPIClient, RequestLimitReached,
                  SchemaError, parse_menu, utc_now, validate_vehicle)
from .utils import load_config, project_root, sha256_file, write_json


def _process_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.WaitForSingleObject.restype = ctypes.c_ulong
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            # ERROR_INVALID_PARAMETER means no such process; access denied is not proof of staleness.
            return ctypes.get_last_error() != 87
        try:
            return kernel.WaitForSingleObject(handle, 0) != 0
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SnapshotLock:
    def __init__(self, snapshot_dir: Path, resume: bool = False) -> None:
        self.path = snapshot_dir / ".collector.lock"
        self.resume = resume

    def __enter__(self):
        if self.path.exists() and self.resume:
            try:
                previous = json.loads(self.path.read_text(encoding="utf-8"))
                if previous.get("host") == socket.gethostname() and not _process_alive(int(previous["pid"])):
                    self.path.unlink()
            except (ValueError, KeyError, json.JSONDecodeError):
                pass
        try:
            with self.path.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "host": socket.gethostname(), "created_at_utc": utc_now()}, stream)
        except FileExistsError as exc:
            raise RuntimeError(f"Collector lock already exists: {self.path}; another collector may be active") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        self.path.unlink(missing_ok=True)


def _unique_ordered(values: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _rotating_pairs(years: list[int], makes_by_year: dict[int, list[str]]) -> list[tuple[int, str]]:
    result = []
    for make_index in range(max((len(makes_by_year[year]) for year in years), default=0)):
        for year in years:
            if make_index < len(makes_by_year[year]):
                result.append((year, makes_by_year[year][make_index]))
    return result


class Collector:
    def __init__(self, root: Path, config: dict[str, Any], snapshot: str, *,
                 config_path: Path | None = None, resume: bool = False,
                 years: list[int] | None = None, makes: list[str] | None = None,
                 max_models_per_make: int | None = None, max_vehicles: int | None = None,
                 max_requests: int | None = None, seed: int | None = None,
                 client_factory: Callable = RawAPIClient, progress: Callable = print) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", snapshot) or snapshot in (".", ".."):
            raise ValueError("snapshot must be a simple directory name, not a path")
        self.root, self.config, self.snapshot = Path(root), config, snapshot
        self.path = self.root / "data" / "raw" / snapshot
        self.config_path, self.resume = config_path, resume
        configured_years = [int(year) for year in config["source"]["years"]]
        self.years = sorted(_unique_ordered(years if years is not None else configured_years))
        if not self.years or any(year not in configured_years for year in self.years):
            raise ValueError("Requested years must be a nonempty subset of configured years")
        self.makes = sorted(_unique_ordered(make.strip() for make in (makes or []) if make.strip())) or None
        for name, value in (("max_models_per_make", max_models_per_make),
                            ("max_vehicles", max_vehicles), ("max_requests", max_requests)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        self.max_models, self.max_vehicles, self.max_requests = max_models_per_make, max_vehicles, max_requests
        self.bounded = any(value is not None for value in (max_models_per_make, max_vehicles, max_requests))
        self.seed = seed if seed is not None else (42 if self.bounded else None)
        self.client_factory, self.progress = client_factory, progress
        self.inventory: dict[str, dict[str, Any]] = {}
        self.state: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.client: RawAPIClient
        self.failures: list[dict[str, Any]] = []
        self.attempted_ids: set[str] = set()
        self.last_report = datetime.now(timezone.utc)

    def _scope(self) -> dict[str, Any]:
        return {"years": self.years, "makes": self.makes, "seed": self.seed,
                "ordering": "seeded_round_robin" if self.seed is not None else "round_robin_sorted"}

    def _prepare(self) -> None:
        if self.resume:
            if not (self.path / "snapshot.json").is_file():
                raise ValueError("--resume needs an existing snapshot.json")
            self.metadata = json.loads((self.path / "snapshot.json").read_text(encoding="utf-8"))
            if self.metadata["requested_scope"] != self._scope():
                raise ValueError("Resume scope/seed differs from this snapshot; use its original arguments or a new snapshot")
            if (self.path / "config.json").is_file():
                frozen_config = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
                if frozen_config != self.config:
                    raise ValueError("Configuration differs from the snapshot's frozen config")
            self.inventory = json.loads((self.path / "inventory.json").read_text(encoding="utf-8"))["records"]
            self.state = json.loads((self.path / "collection_state.json").read_text(encoding="utf-8"))
        else:
            self.metadata = {"schema_version": "1", "snapshot_id": self.snapshot,
                             "started_at_utc": utc_now(), "collection_timezone": "Asia/Qyzylorda",
                             "source_base_url": self.config["source"]["base_url"],
                             "requested_scope": self._scope(), "full_scope_requested": self.years == sorted(self.config["source"]["years"]) and self.makes is None,
                             "config_sha256": sha256_file(self.config_path) if self.config_path else None,
                             "invocations": []}
            self.state = {"completed_models": [], "models_by_pair": {}, "makes_by_year": {},
                          "enumeration_complete": False}
            write_json(self.path / "config.json", self.config)
        self.metadata["invocations"].append({"started_at_utc": utc_now(), "resume": self.resume,
                                              "limits": {"max_models_per_make": self.max_models,
                                                         "max_vehicles_total_in_snapshot": self.max_vehicles,
                                                         "max_requests_this_invocation": self.max_requests}})
        self.metadata.update(status="running", finished_at_utc=None,
                             collection_mode="interleaved_bounded" if self.bounded else "enumerate_then_fetch")
        self._save()
        self.client = self.client_factory(self.path, self.config["source"], max_requests=self.max_requests)
        # Recover vehicle status even if the collector died after writing the API cache.
        for row in self.inventory.values():
            url = self.client._url(f"vehicle/{row['vehicle_id']}", None)
            if url in self.client.entries:
                provenance = row["provenance"][0]
                self.client.cached_payload(url, kind="vehicle", expected=self._expected(row["vehicle_id"], provenance))
                identities = {(p["model_year"], p["manufacturer"], p["model_name"]) for p in row["provenance"]}
                if len(identities) > 1:
                    # A valid record for the original menu does not resolve a later
                    # conflicting discovery. Preserve quarantine across resume.
                    row.update(status="failed", error="Conflicting options-menu identity provenance")
                else:
                    row.update(status="fetched", error=None)
        self._save()

    @staticmethod
    def _expected(vehicle_id: str, provenance: dict[str, Any]) -> dict[str, Any]:
        return {"vehicle_id": vehicle_id, "year": provenance["model_year"],
                "make": provenance["manufacturer"], "model": provenance["model_name"]}

    def _save(self) -> None:
        write_json(self.path / "inventory.json", {"schema_version": "1", "records": self.inventory})
        write_json(self.path / "collection_state.json", self.state)
        temporary = self.path / "inventory.csv.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["vehicle_id", "model_year", "manufacturer", "model_name", "option_text", "status", "error", "provenance_count"])
            writer.writeheader()
            for vehicle_id in sorted(self.inventory, key=int):
                row = self.inventory[vehicle_id]
                primary = row["provenance"][0]
                writer.writerow({"vehicle_id": vehicle_id, **{key: primary[key] for key in ("model_year", "manufacturer", "model_name", "option_text")},
                                 "status": row["status"], "error": row.get("error"), "provenance_count": len(row["provenance"])})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path / "inventory.csv")
        self.metadata.update(last_updated_at_utc=utc_now(), discovered_vehicle_ids=len(self.inventory),
                             fetched_vehicle_records=sum(row["status"] == "fetched" for row in self.inventory.values()),
                             failed_vehicle_records=sum(row["status"] == "failed" for row in self.inventory.values()),
                             enumeration_complete=self.state.get("enumeration_complete", False),
                             completed_model_menus=len(self.state.get("completed_models", [])))
        write_json(self.path / "snapshot.json", self.metadata)

    def _menu(self, name: str, params: dict[str, Any] | None = None) -> list[dict[str, str]]:
        query_hash = hashlib.sha256(json.dumps(params or {}, sort_keys=True).encode()).hexdigest()[:16]
        filename = "menus/year.json" if name == "year" else f"menus/{name}_{query_hash}.json"
        return parse_menu(self.client.fetch_json(f"vehicle/menu/{name}", params=params, file=filename, kind="menu", expected={"menu": name}))

    def _ordered(self, values: Iterable[str], context: str) -> list[str]:
        values = sorted(set(values))
        if self.seed is not None:
            context_seed = hashlib.sha256(f"{self.seed}:{context}".encode()).hexdigest()
            random.Random(context_seed).shuffle(values)
        return values

    def _pairs(self) -> list[tuple[int, str]]:
        available_years = self._menu("year")
        try:
            actual_years = {int(item["value"]) for item in available_years}
        except ValueError as exc:
            raise SchemaError("Year menu contains a noninteger year") from exc
        missing_years = set(self.years) - actual_years
        if missing_years:
            raise SchemaError(f"Requested years missing from source menu: {sorted(missing_years)}")
        makes_by_year = {}
        for year in self.years:
            items = self._menu("make", {"year": year})
            actual_makes = {item["value"] for item in items}
            if self.makes:
                missing = set(self.makes) - actual_makes
                if missing:
                    self.failures.append({"kind": "scope", "model_year": year, "missing_requested_makes": sorted(missing)})
                actual_makes &= set(self.makes)
            makes_by_year[year] = self._ordered(actual_makes, f"makes:{year}")
            self.state["makes_by_year"][str(year)] = makes_by_year[year]
            self._save()
        return _rotating_pairs(self.years, makes_by_year)

    def _models(self, year: int, make: str) -> list[str]:
        key = json.dumps([year, make], ensure_ascii=False)
        if key not in self.state["models_by_pair"]:
            models = self._menu("model", {"year": year, "make": make})
            self.state["models_by_pair"][key] = self._ordered((item["value"] for item in models), f"models:{year}:{make}")
            self._save()
        models = self.state["models_by_pair"][key]
        return models[:self.max_models] if self.max_models is not None else models

    @staticmethod
    def _model_key(year: int, make: str, model: str) -> str:
        return json.dumps([year, make, model], ensure_ascii=False)

    def _discover_model(self, year: int, make: str, model: str) -> None:
        model_key = self._model_key(year, make, model)
        if model_key in self.state["completed_models"]:
            return
        params = {"year": year, "make": make, "model": model}
        items = self._menu("options", params)
        options_url = self.client._url("vehicle/menu/options", params)
        # Validate the entire menu before accepting any ID from it.
        if any(not item["value"].isdigit() or int(item["value"]) <= 0 for item in items):
            raise SchemaError(f"Options menu has invalid vehicle identifiers: {options_url}")
        for item in items:
            vehicle_id = item["value"]
            provenance = {"model_year": year, "manufacturer": make, "model_name": model,
                          "option_text": item["text"], "options_url": options_url}
            row = self.inventory.setdefault(vehicle_id, {"vehicle_id": vehicle_id, "status": "discovered", "error": None, "provenance": []})
            if provenance not in row["provenance"]:
                row["provenance"].append(provenance)
            identities = {(p["model_year"], p["manufacturer"], p["model_name"]) for p in row["provenance"]}
            if len(identities) > 1:
                row.update(status="failed", error="Conflicting options-menu identity provenance")
                self.failures.append({"kind": "provenance", "vehicle_id": vehicle_id, "error": row["error"]})
        self.state["completed_models"].append(model_key)
        self._save()

    def _pending(self, year: int, make: str) -> list[str]:
        ids = [vehicle_id for vehicle_id, row in self.inventory.items()
               if row["status"] != "fetched" and vehicle_id not in self.attempted_ids
               and len({(p["model_year"], p["manufacturer"], p["model_name"]) for p in row["provenance"]}) == 1
               and any(p["model_year"] == year and p["manufacturer"] == make for p in row["provenance"])]
        return self._ordered(ids, f"vehicles:{year}:{make}")

    def _fetched_count(self) -> int:
        return sum(row["status"] == "fetched" for row in self.inventory.values())

    def _download(self, vehicle_id: str) -> None:
        if self.max_vehicles is not None and self._fetched_count() >= self.max_vehicles:
            raise RequestLimitReached(f"Reached --max-vehicles={self.max_vehicles} successful records total")
        row = self.inventory[vehicle_id]
        self.attempted_ids.add(vehicle_id)
        expected = self._expected(vehicle_id, row["provenance"][0])
        try:
            payload = self.client.fetch_json(f"vehicle/{vehicle_id}", file=f"vehicles/{vehicle_id}.json", kind="vehicle", expected=expected)
            for provenance in row["provenance"]:
                validate_vehicle(payload, **self._expected(vehicle_id, provenance))
        except APIError as exc:
            row.update(status="failed", error=str(exc))
            self.failures.append({"kind": "vehicle", "vehicle_id": vehicle_id, "error": str(exc),
                                  "error_category": exc.category, "retryable": exc.retryable, "http_status": exc.status})
        else:
            row.update(status="fetched", error=None)
        self._report()

    def _download_many(self, vehicle_ids: list[str]) -> None:
        """Apply a small batch in discovery order; only this thread mutates inventory."""
        if self.max_vehicles is not None:
            remaining = self.max_vehicles - self._fetched_count()
            if remaining <= 0:
                raise RequestLimitReached(f"Reached --max-vehicles={self.max_vehicles} successful records total")
            vehicle_ids = vehicle_ids[:remaining]
        requests = []
        for vehicle_id in vehicle_ids:
            row = self.inventory[vehicle_id]
            self.attempted_ids.add(vehicle_id)
            requests.append({"endpoint": f"vehicle/{vehicle_id}", "file": f"vehicles/{vehicle_id}.json",
                             "expected": self._expected(vehicle_id, row["provenance"][0])})
        results = self.client.fetch_vehicles(requests)
        budget_error = None
        for vehicle_id, (payload, error) in zip(vehicle_ids, results, strict=True):
            row = self.inventory[vehicle_id]
            if isinstance(error, RequestLimitReached):
                # A request which never started remains eligible on resume.
                self.attempted_ids.discard(vehicle_id)
                budget_error = error
                continue
            if error is not None:
                row.update(status="failed", error=str(error))
                self.failures.append({"kind": "vehicle", "vehicle_id": vehicle_id, "error": str(error),
                                      "error_category": error.category, "retryable": error.retryable, "http_status": error.status})
            else:
                for provenance in row["provenance"]:
                    validate_vehicle(payload, **self._expected(vehicle_id, provenance))
                row.update(status="fetched", error=None)
        self._save()
        self._report()
        if budget_error is not None:
            raise budget_error

    def _report(self, force: bool = False) -> None:
        now = datetime.now(timezone.utc)
        if force or (now - self.last_report).total_seconds() >= 30:
            self.progress(json.dumps({"event": "collection_progress", "snapshot": self.snapshot,
                                      "new_request_attempts": self.client.request_count,
                                      "cache_hits": self.client.cache_hits,
                                      "discovered_ids": len(self.inventory), "fetched_records": self._fetched_count(),
                                      "completed_models": len(self.state["completed_models"]),
                                      "at_utc": utc_now()}, ensure_ascii=False), flush=True)
            self.last_report = now

    def _enumerate_full(self, pairs: list[tuple[int, str]]) -> None:
        queues = {}
        for year, make in pairs:
            try:
                queues[(year, make)] = deque(self._models(year, make))
            except APIError as exc:
                self.failures.append({"kind": "model_menu", "model_year": year, "manufacturer": make, "error": str(exc)})
        while any(queues.values()):
            for (year, make), models in queues.items():
                if not models:
                    continue
                model = models.popleft()
                try:
                    self._discover_model(year, make, model)
                except (APIError, SchemaError) as exc:
                    self.failures.append({"kind": "options_menu", "model_year": year, "manufacturer": make, "model_name": model, "error": str(exc)})
                self._report()
            self._save()
        self.state["enumeration_complete"] = not any(failure["kind"] in ("model_menu", "options_menu", "scope") for failure in self.failures)
        self._save()

    def _fetch_discovered(self, pairs: list[tuple[int, str]]) -> None:
        queues = {(year, make): deque(self._pending(year, make)) for year, make in pairs}
        while any(queues.values()):
            batch = []
            for ids in queues.values():
                if ids:
                    if self.client.workers == 1:
                        self._download(ids.popleft())
                    else:
                        batch.append(ids.popleft())
                        if len(batch) == self.client.workers:
                            self._download_many(batch)
                            batch = []
            if batch:
                self._download_many(batch)
            self._save()

    def _interleave(self, pairs: list[tuple[int, str]]) -> None:
        active = set(pairs)
        attempted_models = set()
        while active:
            batch = []
            for year, make in pairs:
                if (year, make) not in active:
                    continue
                pending = self._pending(year, make)
                if not pending:
                    try:
                        remaining = [model for model in self._models(year, make)
                                     if self._model_key(year, make, model) not in self.state["completed_models"]
                                     and self._model_key(year, make, model) not in attempted_models]
                        if not remaining:
                            active.remove((year, make))
                            continue
                        model = remaining[0]
                        attempted_models.add(self._model_key(year, make, model))
                        self._discover_model(year, make, model)
                    except (APIError, SchemaError) as exc:
                        self.failures.append({"kind": "menu", "model_year": year, "manufacturer": make, "error": str(exc)})
                        # An unsuccessful model menu is not retried endlessly in this invocation.
                        active.remove((year, make))
                        continue
                    pending = self._pending(year, make)
                if pending:
                    if self.client.workers == 1:
                        self._download(pending[0])
                    else:
                        batch.append(pending[0])
                        if len(batch) == self.client.workers:
                            self._download_many(batch)
                            batch = []
            if batch:
                self._download_many(batch)
            self._save()
        self.state["enumeration_complete"] = self.max_models is None and not any(failure["kind"] in ("menu", "scope") for failure in self.failures)
        self._save()

    def run(self) -> dict[str, Any]:
        if self.path.exists() and not self.resume:
            raise ValueError(f"Snapshot already exists; use --resume or choose a new name: {self.path}")
        if not self.path.exists() and self.resume:
            raise ValueError(f"Cannot resume nonexistent snapshot: {self.path}")
        self.path.mkdir(parents=True, exist_ok=True)
        with SnapshotLock(self.path, self.resume):
            try:
                self._prepare()
            except (CorruptCacheError, OSError) as exc:
                # Preparation verifies existing raw evidence before making a
                # request. A failed verification must not leave status=running.
                if self.metadata and self.state:
                    finished = utc_now()
                    self.metadata.update(status="interrupted", finished_at_utc=finished,
                                         stop_reason=f"Preparation failed: {exc}",
                                         full_catalogue_complete=False)
                    if self.metadata.get("invocations"):
                        self.metadata["invocations"][-1].update(finished_at_utc=finished,
                                                                stop_reason=self.metadata["stop_reason"],
                                                                new_request_attempts=0)
                    self._save()
                raise
            stop_reason = None
            error: BaseException | None = None
            try:
                if self.max_vehicles is not None and self._fetched_count() >= self.max_vehicles:
                    stop_reason = "max_vehicles_total_reached"
                else:
                    pairs = self._pairs()
                    if self.bounded:
                        self._interleave(pairs)
                    else:
                        self._enumerate_full(pairs)
                        self._fetch_discovered(pairs)
            except RequestLimitReached as exc:
                stop_reason = str(exc)
            except KeyboardInterrupt as exc:
                stop_reason, error = "interrupted_by_user", exc
            except (APIError, SchemaError, CorruptCacheError, OSError) as exc:
                stop_reason, error = str(exc), exc
                self.failures.append({"kind": "fatal", "error": str(exc), "type": type(exc).__name__})
            finally:
                finished = utc_now()
                failures_path = self.path / "collection_failures.jsonl"
                if self.failures:
                    with failures_path.open("a", encoding="utf-8") as stream:
                        for failure in self.failures:
                            stream.write(json.dumps({**failure, "at_utc": finished}, ensure_ascii=False) + "\n")
                all_fetched = all(row["status"] == "fetched" for row in self.inventory.values())
                complete = self.state["enumeration_complete"] and all_fetched and not self.failures and stop_reason is None
                self.metadata.update(status="complete" if complete else ("interrupted" if error else "partial"),
                                     finished_at_utc=finished, stop_reason=stop_reason,
                                     collection_complete_for_discovered_ids=all_fetched,
                                     full_catalogue_complete=complete and self.metadata["full_scope_requested"],
                                     request_attempts_total=sum(1 for _ in self.client.manifest_path.open(encoding="utf-8")) if self.client.manifest_path.exists() else 0,
                                     successful_http_responses=len(self.client.entries))
                self.metadata["invocations"][-1].update(finished_at_utc=finished, new_request_attempts=self.client.request_count,
                                                        cache_hits=self.client.cache_hits, stop_reason=stop_reason)
                self._save()
                self._report(force=True)
            if error is not None:
                raise error
            return self.metadata


def _comma_years(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integer model years") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/project.json"))
    parser.add_argument("--snapshot", required=True, help="New raw snapshot directory name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--years", type=_comma_years, help="Comma-separated subset of configured model years")
    parser.add_argument("--makes", help="Comma-separated exact API manufacturer names")
    parser.add_argument("--max-models-per-make", type=int, help="Model cap per year/manufacturer, ordered independently of target")
    parser.add_argument("--max-vehicles", type=int, help="Total successful individual records in this snapshot, including previous invocations")
    parser.add_argument("--max-requests", type=int, help="New HTTP attempts this invocation, including retries")
    parser.add_argument("--seed", type=int, help="Deterministic target-independent ordering; bounded default 42")
    args = parser.parse_args(argv)
    try:
        config_path = args.config.resolve()
        collector = Collector(project_root(), load_config(config_path), args.snapshot,
                              config_path=config_path, resume=args.resume, years=args.years,
                              makes=args.makes.split(",") if args.makes else None,
                              max_models_per_make=args.max_models_per_make, max_vehicles=args.max_vehicles,
                              max_requests=args.max_requests, seed=args.seed)
        metadata = collector.run()
    except KeyboardInterrupt:
        print("Collection interrupted; durable snapshot can be resumed.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"event": "collection_finished", "snapshot": args.snapshot,
                      "status": metadata["status"], "fetched_vehicle_records": metadata["fetched_vehicle_records"],
                      "full_catalogue_complete": metadata["full_catalogue_complete"],
                      "stop_reason": metadata["stop_reason"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
