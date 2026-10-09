"""Small, polite FuelEconomy.gov client with an immutable response cache.

Domain eligibility belongs to ``clean``. This module validates the transport,
the documented menu shape, and vehicle identity against discovery provenance.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Mapping

from .utils import write_json


class SchemaError(ValueError):
    """The response is JSON, but is not the expected API resource."""


class CorruptCacheError(RuntimeError):
    """Cached evidence no longer matches its original recorded checksum."""


class RequestLimitReached(RuntimeError):
    """The invocation's explicit network-attempt budget has been reached."""


class APIError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None,
                 retryable: bool = False, category: str = "http") -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.category = category


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_menu(payload: Any) -> list[dict[str, str]]:
    """Normalize documented None/single/list menu items; reject malformed JSON."""
    if payload is None:
        return []
    if not isinstance(payload, dict) or "menuItem" not in payload:
        raise SchemaError("Expected a menu object with a menuItem property")
    items = payload["menuItem"]
    if items is None:
        return []
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        raise SchemaError("menuItem must be null, an object, or a list")
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            raise SchemaError("Every menuItem must be an object")
        row = {}
        for key in ("text", "value"):
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise SchemaError(f"Menu {key} must be a nonempty string or integer")
            value = str(value).strip()
            if not value:
                raise SchemaError(f"Menu {key} must not be empty")
            row[key] = value
        normalized.append(row)
    return normalized


def validate_vehicle(payload: Any, *, vehicle_id: str | int | None = None,
                     year: int | None = None, make: str | None = None,
                     model: str | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SchemaError("Expected an individual vehicle JSON object")
    actual_id = payload.get("id")
    if isinstance(actual_id, bool) or not isinstance(actual_id, (str, int)):
        raise SchemaError("Vehicle id must be a positive integer identifier")
    actual_id = str(actual_id)
    if not actual_id.isdigit() or int(actual_id) <= 0:
        raise SchemaError("Vehicle id must be a positive integer identifier")
    actual_year = payload.get("year")
    if isinstance(actual_year, bool) or not isinstance(actual_year, (str, int)):
        raise SchemaError("Vehicle year must be an integer")
    try:
        actual_year = int(actual_year)
    except (ValueError, TypeError) as exc:
        raise SchemaError("Vehicle year must be an integer") from exc
    if not 1900 <= actual_year <= 2100:
        raise SchemaError("Vehicle year is outside the catalogue's plausible range")
    for key in ("make", "model"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise SchemaError(f"Vehicle {key} must be a nonempty string")
    checks = (("id", actual_id, None if vehicle_id is None else str(vehicle_id)),
              ("year", actual_year, year), ("make", payload["make"], make),
              ("model", payload["model"], model))
    for key, actual, expected in checks:
        if expected is not None and actual != expected:
            raise SchemaError(f"Vehicle {key} disagrees with its options menu: {actual!r} != {expected!r}")
    return payload


def default_transport(url: str, headers: Mapping[str, str], timeout: float) -> HTTPResponse:
    request = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HTTPResponse(response.status, dict(response.headers), response.read())
    except urllib.error.HTTPError as exc:
        # HTTPError is still an HTTP response, whose exact bytes are evidence.
        return HTTPResponse(exc.code, dict(exc.headers or {}), exc.read())


def _safe_relative(path: str | Path) -> str:
    path = Path(path)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("Response filenames must be relative paths inside the snapshot")
    return path.as_posix()


def _immutable_bytes(path: Path, body: bytes) -> None:
    """Flush a new response then atomically publish it; never replace evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CorruptCacheError(f"Refusing to overwrite existing raw evidence: {path}")
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise CorruptCacheError(f"Raw evidence appeared while writing: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class RawAPIClient:
    """An immutable-evidence client. Callers hold the snapshot collector lock.

    ``max_requests`` counts new network attempts in this invocation, including
    retries. Cached reads do not consume that budget. A pending journal closes
    the body/manifest crash window; checksums are always rechecked on reuse.
    """
    def __init__(self, snapshot_dir: Path, source_config: Mapping[str, Any], *,
                 transport: Callable = default_transport, monotonic: Callable = time.monotonic,
                 sleep: Callable = time.sleep, now: Callable = utc_now,
                 rng: random.Random | None = None, max_requests: int | None = None) -> None:
        self.root = Path(snapshot_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.source = dict(source_config)
        self.base_url = self.source["base_url"].rstrip("/") + "/"
        self.min_interval = float(self.source.get("min_interval_seconds", 1.0))
        self.timeout = float(self.source.get("timeout_seconds", 30))
        self.max_attempts = int(self.source.get("max_attempts", 5))
        self.workers = int(self.source.get("workers", 1))
        if not 1 <= self.workers <= 4:
            raise ValueError("workers must be between one and four")
        if not math.isfinite(self.min_interval) or self.min_interval < 1 / self.workers:
            raise ValueError("min_interval_seconds must be at least 1 / workers")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 1 <= self.max_attempts <= 5:
            raise ValueError("Collector supports at most five attempts")
        if max_requests is not None and max_requests < 0:
            raise ValueError("max_requests must be nonnegative")
        self.transport, self.monotonic, self.sleep, self.now = transport, monotonic, sleep, now
        self.rng = rng or random.Random()
        self.max_requests, self.request_count, self.cache_hits = max_requests, 0, 0
        self.last_start: float | None = None
        self._admission_lock = threading.Lock()
        self._not_before = 0.0
        self.index_path = self.root / "cache_index.json"
        self.manifest_path = self.root / "requests.jsonl"
        self.entries: dict[str, dict[str, Any]] = {}
        self._recover_and_index()

    def _append_manifest(self, entry: dict[str, Any]) -> None:
        with self.manifest_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _save_index(self) -> None:
        # This growing lookup table is a derived cache, not primary evidence.
        # Compact dumps uses the fast encoder and one write; pretty json.dump
        # emits thousands of tiny writes for every checkpoint at catalogue scale.
        body = json.dumps({"schema_version": "1", "entries": self.entries}, ensure_ascii=False,
                          allow_nan=False, default=str, separators=(",", ":")) + "\n"
        temporary = None
        try:
            with NamedTemporaryFile("w", encoding="utf-8", dir=self.root,
                                    prefix=self.index_path.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.index_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _manifest_entries(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        entries = []
        with self.manifest_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CorruptCacheError(f"Invalid requests.jsonl line {line_number}; preserve evidence and inspect") from exc
                if not isinstance(entry, dict):
                    raise CorruptCacheError(f"Invalid requests.jsonl entry {line_number}")
                entries.append(entry)
        return entries

    def _recover_and_index(self) -> None:
        entries = self._manifest_entries()
        completed = {entry.get("request_id") for entry in entries}
        journal_dir = self.root / "pending_requests"
        for journal in sorted(journal_dir.glob("*.json")) if journal_dir.exists() else []:
            entry = json.loads(journal.read_text(encoding="utf-8"))
            if entry.get("request_id") not in completed:
                raw_path = self.root / _safe_relative(entry["file"])
                if raw_path.exists():
                    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != entry["sha256"]:
                        raise CorruptCacheError(f"Interrupted response checksum mismatch: {raw_path}")
                    entry["recovered_after_interruption"] = True
                else:
                    entry.update(file=None, sha256=None, valid=False,
                                 error="Interrupted before publishing response bytes",
                                 error_category="interrupted", retryable=True)
                self._append_manifest(entry)
                entries.append(entry)
            journal.unlink()
        for entry in entries:
            if entry.get("valid") is True and entry.get("status") == 200:
                url = entry.get("url")
                if not url or not entry.get("file") or not entry.get("sha256"):
                    raise CorruptCacheError("A successful manifest entry has incomplete provenance")
                previous = self.entries.get(url)
                if previous and (previous["sha256"], previous["file"]) != (entry["sha256"], entry["file"]):
                    raise CorruptCacheError(f"Conflicting immutable responses for {url}")
                self.entries[url] = entry
        # Completed enumeration is represented in collector state, so some menus
        # will not be reread by that layer on resume. Verify every successful
        # response here before trusting any state that depends on the cache.
        for url, entry in self.entries.items():
            resource_kind = entry.get("kind")
            if resource_kind not in ("menu", "vehicle"):
                raise CorruptCacheError(f"Successful cache lacks a resource kind: {url}")
            self.cached_payload(url, kind=resource_kind, expected=entry.get("expected"))
        # Manifest is authoritative; rebuilding also recovers an interrupted index write.
        self._save_index()

    def _url(self, endpoint: str, params: Mapping[str, Any] | None) -> str:
        if "://" in endpoint or endpoint.startswith("/") or ".." in Path(endpoint).parts:
            raise ValueError("Use a relative, documented API endpoint")
        url = self.base_url + endpoint
        if params:
            url += "?" + urllib.parse.urlencode(params)
        return url

    @staticmethod
    def _validate(body: bytes, kind: str, expected: Mapping[str, Any] | None) -> Any:
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SchemaError("Expected UTF-8 JSON, received a different or malformed body") from exc
        if kind == "menu":
            items = parse_menu(payload)
            menu_name = (expected or {}).get("menu")
            if menu_name in ("year", "options"):
                for item in items:
                    if not item["value"].isdigit() or int(item["value"]) <= 0:
                        raise SchemaError(f"{menu_name} menu must contain positive integer values")
        elif kind == "vehicle":
            validate_vehicle(payload, **dict(expected or {}))
        else:
            raise ValueError(f"Unknown resource kind: {kind}")
        return payload

    def cached_payload(self, url: str, *, kind: str, expected: Mapping[str, Any] | None = None) -> Any | None:
        entry = self.entries.get(url)
        if entry is None:
            return None
        path = self.root / _safe_relative(entry["file"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise CorruptCacheError(f"Checksum mismatch or missing successful cache: {path}; use a new snapshot")
        try:
            return self._validate(path.read_bytes(), kind, expected)
        except SchemaError as exc:
            raise CorruptCacheError(f"Successful cache failed current schema validation: {path}: {exc}") from exc

    def _retry_after(self, headers: Mapping[str, str]) -> float | None:
        value = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
        if value is None or not self.source.get("respect_retry_after", True):
            return None
        try:
            seconds = float(value)
            return max(0.0, seconds) if math.isfinite(seconds) else None
        except (ValueError, TypeError):
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                return max(0.0, (date - datetime.fromisoformat(self.now())).total_seconds())
            except (ValueError, TypeError, OverflowError):
                return None

    def _batch_attempt(self, request: Mapping[str, Any], attempt: int) -> tuple[dict[str, Any], HTTPResponse | None, Any, APIError | None]:
        """Network worker: coordinate starts and return evidence, never write files."""
        url = request["url"]
        while True:
            with self._admission_lock:
                if self.max_requests is not None and self.request_count >= self.max_requests:
                    raise RequestLimitReached(f"Reached --max-requests={self.max_requests}")
                earliest = max(self._not_before, (self.last_start + self.min_interval) if self.last_start is not None else 0.0)
                wait = earliest - self.monotonic()
                if wait <= 0:
                    self.last_start = self.monotonic()
                    self.request_count += 1
                    requested_at = self.now()
                    break
            # Sleeping outside the lock lets an in-flight 429 response install
            # its global cooldown immediately. Recheck after every wake-up.
            self.sleep(wait)
        entry = {"request_id": uuid.uuid4().hex, "url": url, "params": dict(request.get("params") or {}),
                 "requested_at_utc": requested_at, "requested_accept": "application/json",
                 "attempt": attempt, "status": None, "valid": False, "file": None,
                 "sha256": None, "kind": request["kind"], "expected": dict(request.get("expected") or {})}
        response, payload, error = None, None, None
        try:
            response = self.transport(url, {"Accept": "application/json", "User-Agent": "SE-2421-fuel-consumption-research/0.1"}, self.timeout)
            if response.status != 200:
                raise APIError(f"HTTP {response.status} for {url}", status=response.status,
                               retryable=response.status in (408, 429) or 500 <= response.status <= 599)
            try:
                payload = self._validate(response.body, request["kind"], request.get("expected"))
            except SchemaError as exc:
                raise APIError(str(exc), status=200, category="schema") from exc
            entry["valid"] = True
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = APIError(f"Network failure for {url}: {exc}", retryable=True, category="network")
        except APIError as exc:
            error = exc
        entry["received_at_utc"] = self.now()
        if error is not None:
            entry.update(error=str(error), error_category=error.category, retryable=error.retryable)
        # Retry-After applies to every later request, including another resource kind.
        if response is not None and response.status != 200:
            retry_after = self._retry_after(response.headers)
            if retry_after is not None:
                with self._admission_lock:
                    self._not_before = max(self._not_before, self.monotonic() + retry_after)
        return entry, response, payload, error

    def _commit_batch_attempt(self, request: Mapping[str, Any], entry: dict[str, Any], response: HTTPResponse | None) -> None:
        """Main-thread-only durable publication, with the same crash journal as fetch_json."""
        if response is not None:
            destination = request["file"] if entry["valid"] else f"failures/{entry['request_id']}.body"
            entry.update(status=response.status, response_headers=dict(response.headers),
                         content_type=next((v for k, v in response.headers.items() if k.lower() == "content-type"), None),
                         file=destination, sha256=hashlib.sha256(response.body).hexdigest(), byte_count=len(response.body))
            journal = self.root / "pending_requests" / f"{entry['request_id']}.json"
            write_json(journal, entry)
            _immutable_bytes(self.root / destination, response.body)
            self._append_manifest(entry)
            journal.unlink()
        else:
            self._append_manifest(entry)
        if entry["valid"]:
            self.entries[request["url"]] = entry

    def fetch_vehicles(self, requests: list[Mapping[str, Any]]) -> list[tuple[Any, APIError | RequestLimitReached | None]]:
        """Fetch at most ``workers`` individual records, returning results in input order.

        Each request has endpoint, file, and optional expected identity. All raw
        bytes and successful responses are committed by the calling thread;
        worker threads only perform bounded HTTP attempts. Retries form waves,
        with a shared cooldown. Failed records remain separate from valid cache.
        """
        return self._fetch_batch(requests, kind="vehicle")

    def fetch_menus(self, requests: list[Mapping[str, Any]]) -> list[tuple[Any, APIError | RequestLimitReached | None]]:
        """Fetch documented menus in input order, sharing rate, retries and writer.

        Each request has endpoint, params and file. The endpoint supplies the
        expected menu name; an explicitly conflicting name is rejected before
        any network attempt. JSON null remains a valid cached empty menu.
        """
        return self._fetch_batch(requests, kind="menu")

    def _fetch_batch(self, requests: list[Mapping[str, Any]], *, kind: str) -> list[tuple[Any, APIError | RequestLimitReached | None]]:
        if not requests or len(requests) > self.workers:
            raise ValueError(f"A {kind} batch must contain between one and workers requests")
        prepared = []
        urls, filenames = set(), set()
        results: list[tuple[Any, APIError | RequestLimitReached | None] | None] = [None] * len(requests)
        for position, item in enumerate(requests):
            endpoint = item["endpoint"]
            expected = dict(item.get("expected") or {})
            if kind == "vehicle":
                if not endpoint.startswith("vehicle/") or not endpoint.removeprefix("vehicle/").isdigit():
                    raise ValueError("Batches support individual vehicle endpoints only")
                params = None  # Preserve the individual-record wrapper's original contract.
            else:
                menu_name = endpoint.removeprefix("vehicle/menu/")
                if not endpoint.startswith("vehicle/menu/") or menu_name not in ("year", "make", "model", "options"):
                    raise ValueError("Menu batches support documented year, make, model and options endpoints only")
                if expected.get("menu", menu_name) != menu_name:
                    raise ValueError("Expected menu name disagrees with its endpoint")
                expected["menu"] = menu_name
                params = dict(item.get("params") or {})
            request = {"url": self._url(endpoint, params), "file": _safe_relative(item["file"]),
                       "kind": kind, "params": dict(params or {}),
                       "expected": expected, "position": position}
            if request["url"] in urls or request["file"] in filenames:
                raise ValueError("A batch must contain distinct URLs and raw filenames")
            urls.add(request["url"])
            filenames.add(request["file"])
            if request["url"] in self.entries:
                results[position] = (self.cached_payload(request["url"], kind=kind, expected=request["expected"]), None)
                self.cache_hits += 1
            elif (self.root / request["file"]).exists():
                raise CorruptCacheError(f"Unindexed raw evidence already exists: {request['file']}")
            else:
                prepared.append(request)
        pending = prepared
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="epa-http") as executor:
            for attempt in range(1, self.max_attempts + 1):
                futures = [(request, executor.submit(self._batch_attempt, request, attempt)) for request in pending]
                retry = []
                fatal = None
                entries_before = len(self.entries)
                try:
                    for request, future in futures:
                        try:
                            entry, response, payload, error = future.result()
                        except RequestLimitReached as exc:
                            results[request["position"]] = (None, exc)
                            continue
                        except BaseException as exc:
                            # Drain and commit the other completed responses before
                            # propagating an interruption or unexpected worker failure.
                            fatal = fatal or exc
                            continue
                        self._commit_batch_attempt(request, entry, response)
                        if error is None:
                            results[request["position"]] = (payload, None)
                        elif error.retryable and attempt < self.max_attempts:
                            retry.append(request)
                        else:
                            results[request["position"]] = (None, error)
                finally:
                    # Primary raw+manifest evidence is already durable per response.
                    # Publish the derived lookup once per wave, including partial
                    # successes before a writer error or worker interruption.
                    if len(self.entries) != entries_before:
                        self._save_index()
                if fatal is not None:
                    raise fatal
                if not retry:
                    break
                # A conservative batch-wide backoff prevents another vehicle
                # from bypassing a retry pause. Jitter is sampled by the writer.
                backoff = 2 ** (attempt - 1) + self.rng.uniform(0, 1)
                with self._admission_lock:
                    self._not_before = max(self._not_before, self.monotonic() + backoff)
                pending = retry
        assert all(result is not None for result in results)
        return results  # type: ignore[return-value]

    def fetch_json(self, endpoint: str, *, params: Mapping[str, Any] | None = None,
                   file: str | Path, kind: str, expected: Mapping[str, Any] | None = None) -> Any:
        url = self._url(endpoint, params)
        relative_file = _safe_relative(file)
        # A null JSON menu is legitimate, so membership (rather than payload != None) detects a hit.
        if url in self.entries:
            payload = self.cached_payload(url, kind=kind, expected=expected)
            self.cache_hits += 1
            return payload
        if (self.root / relative_file).exists():
            raise CorruptCacheError(f"Unindexed raw evidence already exists: {relative_file}")
        headers = {"Accept": "application/json", "User-Agent": "SE-2421-fuel-consumption-research/0.1"}
        last_error: APIError | None = None
        for attempt in range(1, self.max_attempts + 1):
            if self.max_requests is not None and self.request_count >= self.max_requests:
                raise RequestLimitReached(f"Reached --max-requests={self.max_requests}")
            earliest = max(self._not_before, (self.last_start + self.min_interval) if self.last_start is not None else 0.0)
            wait = earliest - self.monotonic()
            if wait > 0:
                self.sleep(wait)
            self.last_start = self.monotonic()
            self.request_count += 1
            entry = {"request_id": uuid.uuid4().hex, "url": url, "params": dict(params or {}),
                     "requested_at_utc": self.now(), "requested_accept": "application/json",
                     "attempt": attempt, "status": None, "valid": False, "file": None,
                     "sha256": None, "kind": kind, "expected": dict(expected or {})}
            response: HTTPResponse | None = None
            payload = None
            try:
                response = self.transport(url, headers, self.timeout)
                if response.status != 200:
                    retryable = response.status in (408, 429) or 500 <= response.status <= 599
                    raise APIError(f"HTTP {response.status} for {url}", status=response.status,
                                   retryable=retryable)
                try:
                    payload = self._validate(response.body, kind, expected)
                except SchemaError as exc:
                    raise APIError(str(exc), status=200, category="schema") from exc
                entry["valid"] = True
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = APIError(f"Network failure for {url}: {exc}", retryable=True, category="network")
            except APIError as exc:
                last_error = exc
            entry["received_at_utc"] = self.now()
            if not entry["valid"]:
                assert last_error is not None
                entry.update(error=str(last_error), error_category=last_error.category,
                             retryable=last_error.retryable)
            if response is not None:
                destination = relative_file if entry["valid"] else f"failures/{entry['request_id']}.body"
                entry.update(status=response.status, response_headers=dict(response.headers),
                             content_type=next((v for k, v in response.headers.items() if k.lower() == "content-type"), None),
                             file=destination, sha256=hashlib.sha256(response.body).hexdigest(),
                             byte_count=len(response.body))
                journal = self.root / "pending_requests" / f"{entry['request_id']}.json"
                write_json(journal, entry)
                _immutable_bytes(self.root / destination, response.body)
                self._append_manifest(entry)
                journal.unlink()
            else:
                self._append_manifest(entry)
            if entry["valid"]:
                self.entries[url] = entry
                self._save_index()
                return payload
            assert last_error is not None
            if not last_error.retryable or attempt == self.max_attempts:
                raise last_error
            backoff = 2 ** (attempt - 1) + self.rng.uniform(0, 1)
            retry_after = self._retry_after(response.headers) if response else None
            self.sleep(max(backoff, retry_after or 0.0))
        assert last_error is not None
        raise last_error
