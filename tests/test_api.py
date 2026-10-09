"""Offline behavior checks; no test accesses the public source."""
import hashlib
import json
import random
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fuel_consumption.api import (APIError, CorruptCacheError, HTTPResponse,
                                 RawAPIClient, RequestLimitReached, SchemaError,
                                 parse_menu, validate_vehicle)
from fuel_consumption.utils import write_json


class FakeClock:
    def __init__(self):
        self.seconds = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.seconds

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.seconds += seconds

    def now(self):
        return (datetime(2026, 10, 9, tzinfo=timezone.utc) + timedelta(seconds=self.seconds)).isoformat()


def source_config():
    return {"base_url": "https://www.fueleconomy.gov/ws/rest/", "min_interval_seconds": 1,
            "timeout_seconds": 30, "max_attempts": 5, "workers": 1, "respect_retry_after": True}


def response(payload, status=200, headers=None):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HTTPResponse(status, headers or {"Content-Type": "application/json"}, body)


def client(tmp_path, transport, clock=None, **kwargs):
    clock = clock or FakeClock()
    return RawAPIClient(tmp_path, source_config(), transport=transport,
                        monotonic=clock.monotonic, sleep=clock.sleep,
                        now=clock.now, rng=random.Random(42), **kwargs)


@pytest.mark.parametrize("payload", [None, {"menuItem": None}, {"menuItem": []}])
def test_empty_menu_shapes(payload):
    assert parse_menu(payload) == []


def test_single_and_list_menu_real_fixture():
    pilot = Path(__file__).resolve().parents[1] / "evidence/api_probe"
    single = json.loads((pilot / "options_2025_Toyota_Prius.json").read_text())
    expected = [{"text": "Auto (variable gear ratios), 4 cyl, 2.0 L", "value": "48861"}]
    assert parse_menu(single) == expected
    assert parse_menu({"menuItem": [single["menuItem"]]}) == expected


@pytest.mark.parametrize("payload", [{}, [], "html", {"menuItem": "bad"},
                                      {"menuItem": [None]}, {"menuItem": {"text": "x"}},
                                      {"menuItem": {"text": "x", "value": True}}])
def test_malformed_menus_are_not_silent_empty(payload):
    with pytest.raises(SchemaError):
        parse_menu(payload)


def test_real_vehicle_identity_and_conflicts():
    pilot = Path(__file__).resolve().parents[1] / "evidence/api_probe"
    payload = json.loads((pilot / "vehicle_34836.json").read_text())
    assert validate_vehicle(payload, vehicle_id="34836", year=2015, make="Honda", model="Fit") is payload
    with pytest.raises(SchemaError, match="disagrees"):
        validate_vehicle(payload, vehicle_id="34836", year=2025)


def test_exact_bytes_checksum_throttle_and_resume(tmp_path):
    clock = FakeClock()
    starts = []
    body = b'{ "menuItem": { "text": "Honda", "value": "Honda" } }\n'
    def transport(url, headers, timeout):
        starts.append(clock.seconds)
        assert headers["Accept"] == "application/json"
        assert timeout == 30
        return response(body)
    api = client(tmp_path, transport, clock)
    for year in (2015, 2016):
        api.fetch_json("vehicle/menu/make", params={"year": year}, file=f"menus/{year}.json", kind="menu")
    assert starts == [0.0, 1.0]
    assert (tmp_path / "menus/2015.json").read_bytes() == body
    lines = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert lines[0]["sha256"] == hashlib.sha256(body).hexdigest()
    assert lines[0]["valid"] is True and lines[0]["status"] == 200
    assert lines[0]["file"] == "menus/2015.json"
    resumed = client(tmp_path, lambda *args: pytest.fail("Valid cache must not be requested again"))
    resumed.fetch_json("vehicle/menu/make", params={"year": 2015}, file="menus/2015.json", kind="menu")
    assert resumed.request_count == 0 and resumed.cache_hits == 1


def test_retry_after_retryable_http_and_network_failures(tmp_path):
    clock = FakeClock()
    events = [urllib.error.URLError("offline"), response(b"busy", 429, {"Retry-After": "7"}),
              response(b"later", 503), response({"menuItem": []})]
    def transport(*args):
        event = events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event
    api = client(tmp_path, transport, clock)
    assert api.fetch_json("vehicle/menu/year", file="menus/year.json", kind="menu") == {"menuItem": []}
    assert api.request_count == 4
    assert 7.0 in clock.sleeps
    lines = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert [row["status"] for row in lines] == [None, 429, 503, 200]
    assert lines[0]["error_category"] == "network"
    assert (tmp_path / lines[1]["file"]).read_bytes() == b"busy"


def test_retry_after_http_date(tmp_path):
    clock = FakeClock()
    events = [response(b"busy", 429, {"Retry-After": "Fri, 09 Oct 2026 00:00:10 GMT"}),
              response({"menuItem": []})]
    api = client(tmp_path, lambda *args: events.pop(0), clock)
    api.fetch_json("vehicle/menu/year", file="menus/year.json", kind="menu")
    assert clock.sleeps == [10.0]


def test_nonretryable_and_invalid_json_logged(tmp_path):
    calls = []
    api = client(tmp_path, lambda *args: (calls.append(args) or response(b"missing", 404)))
    with pytest.raises(APIError) as failure:
        api.fetch_json("vehicle/999", file="vehicles/999.json", kind="vehicle")
    assert failure.value.retryable is False and len(calls) == 1
    html_api = client(tmp_path / "html", lambda *args: response(b"<html>Error</html>"))
    with pytest.raises(APIError) as failure:
        html_api.fetch_json("vehicle/123", file="vehicles/123.json", kind="vehicle")
    assert failure.value.category == "schema"
    assert not (tmp_path / "html/vehicles/123.json").exists()


def test_options_ids_schema_is_validated_before_success(tmp_path):
    api = client(tmp_path, lambda *args: response({"menuItem": {"text": "engine", "value": "invalid"}}))
    with pytest.raises(APIError, match="positive integer"):
        api.fetch_json("vehicle/menu/options", file="menus/options.json", kind="menu", expected={"menu": "options"})
    assert not api.entries


def test_attempts_are_bounded_and_budget_includes_retries(tmp_path):
    api = client(tmp_path, lambda *args: response(b"later", 503))
    with pytest.raises(APIError):
        api.fetch_json("vehicle/menu/year", file="menus/year.json", kind="menu")
    assert api.request_count == 5
    limited = client(tmp_path / "budget", lambda *args: response(b"later", 503), max_requests=2)
    with pytest.raises(RequestLimitReached):
        limited.fetch_json("vehicle/menu/year", file="menus/year.json", kind="menu")
    assert limited.request_count == 2


def test_corrupted_successful_cache_is_refused_without_overwrite(tmp_path):
    api = client(tmp_path, lambda *args: response({"menuItem": []}))
    api.fetch_json("vehicle/menu/year", file="menus/year.json", kind="menu")
    (tmp_path / "menus/year.json").write_bytes(b'{"menuItem":null}')
    with pytest.raises(CorruptCacheError, match="Checksum"):
        client(tmp_path, lambda *args: pytest.fail("Do not overwrite corrupt raw evidence"))
    assert (tmp_path / "menus/year.json").read_bytes() == b'{"menuItem":null}'


def test_interrupted_body_manifest_window_recovers_with_original_hash(tmp_path):
    body = b'{"menuItem":[]}'
    raw = tmp_path / "menus/year.json"
    raw.parent.mkdir()
    raw.write_bytes(body)
    url = source_config()["base_url"] + "vehicle/menu/year"
    entry = {"request_id": "interrupted", "url": url, "file": "menus/year.json", "status": 200,
             "valid": True, "kind": "menu", "expected": {}, "sha256": hashlib.sha256(body).hexdigest(),
             "received_at_utc": "2026-10-09T00:00:00+00:00"}
    write_json(tmp_path / "pending_requests/interrupted.json", entry)
    api = client(tmp_path, lambda *args: pytest.fail("Recovered bytes must be reused"))
    assert api.fetch_json("vehicle/menu/year", file="menus/year.json", kind="menu") == {"menuItem": []}
    assert not (tmp_path / "pending_requests/interrupted.json").exists()
    assert json.loads((tmp_path / "requests.jsonl").read_text())["recovered_after_interruption"] is True


@pytest.mark.parametrize("config_change", [{"min_interval_seconds": 0.1}, {"workers": 5}, {"workers": 0},
                                          {"workers": 4, "min_interval_seconds": 0.1}, {"max_attempts": 6}])
def test_polite_collection_settings_cannot_be_weakened(tmp_path, config_change):
    with pytest.raises(ValueError):
        RawAPIClient(tmp_path, {**source_config(), **config_change})


def batch_requests(ids=(1, 2, 3, 4)):
    return [{"endpoint": f"vehicle/{vehicle_id}", "file": f"vehicles/{vehicle_id}.json",
             "expected": {"vehicle_id": str(vehicle_id), "year": 2025, "make": "Honda", "model": "Civic"}}
            for vehicle_id in ids]


def batch_response(vehicle_id):
    return response({"id": str(vehicle_id), "year": "2025", "make": "Honda", "model": "Civic"})


def batch_client(tmp_path, transport, clock=None, **kwargs):
    config = {**source_config(), "workers": 4, "min_interval_seconds": .25}
    if clock is None:
        return RawAPIClient(tmp_path, config, transport=transport, rng=random.Random(42), **kwargs)
    return RawAPIClient(tmp_path, config, transport=transport, monotonic=clock.monotonic,
                        sleep=clock.sleep, now=clock.now, rng=random.Random(42), **kwargs)


def test_batch_overlaps_requests_global_rate_and_single_writer(tmp_path):
    import threading
    import time
    lock = threading.Lock()
    starts, active, peak = [], 0, 0
    def transport(url, headers, timeout):
        nonlocal active, peak
        with lock:
            starts.append(time.monotonic())
            active += 1
            peak = max(peak, active)
        time.sleep(.8)
        with lock:
            active -= 1
        return batch_response(url.rsplit("/", 1)[-1])
    api = batch_client(tmp_path, transport)
    writer_threads = []
    append_manifest = api._append_manifest
    def observed_write(entry):
        writer_threads.append(threading.get_ident())
        append_manifest(entry)
    api._append_manifest = observed_write
    results = api.fetch_vehicles(batch_requests())
    assert peak >= 2 and peak <= 4
    assert all(b - a >= .23 for a, b in zip(starts, starts[1:]))
    assert [payload["id"] for payload, error in results] == ["1", "2", "3", "4"]
    assert all(error is None for _, error in results)
    assert writer_threads == [threading.get_ident()] * 4
    for entry in api.entries.values():
        assert hashlib.sha256((tmp_path / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
    resumed = batch_client(tmp_path, lambda *args: pytest.fail("Resume must use immutable cache"))
    assert resumed.fetch_vehicles(batch_requests()) == results
    assert resumed.request_count == 0 and resumed.cache_hits == 4


def test_batch_retry_after_pauses_every_later_admission_and_logs_failure(tmp_path):
    import threading
    clock = FakeClock()
    observed = []
    lock = threading.Lock()
    attempts = {}
    def transport(url, headers, timeout):
        vehicle_id = url.rsplit("/", 1)[-1]
        with lock:
            observed.append((vehicle_id, clock.seconds))
            attempts[vehicle_id] = attempts.get(vehicle_id, 0) + 1
            if vehicle_id == "1" and attempts[vehicle_id] == 1:
                return response(b"busy", 429, {"Retry-After": "7"})
        return batch_response(vehicle_id)
    api = batch_client(tmp_path, transport, clock)
    results = api.fetch_vehicles(batch_requests())
    assert all(error is None for _, error in results)
    assert observed[0] == ("1", 0.0)
    assert all(start >= 7 for _, start in observed[1:])
    assert api.request_count == 5
    lines = [json.loads(line) for line in api.manifest_path.read_text().splitlines()]
    failed = next(row for row in lines if row["status"] == 429)
    assert failed["retryable"] and (tmp_path / failed["file"]).read_bytes() == b"busy"
    assert all(row["kind"] == "vehicle" for row in lines)


def test_batch_inflight_retry_after_interrupts_other_workers_waiting_for_slots(tmp_path):
    import threading
    import time
    lock = threading.Lock()
    starts = {}
    first_response_at = []
    def transport(url, *args):
        vehicle_id = url.rsplit("/", 1)[-1]
        with lock:
            starts[vehicle_id] = time.monotonic()
        if vehicle_id == "1":
            time.sleep(.1)
            first_response_at.append(time.monotonic())
            return response(b"busy", 429, {"Retry-After": ".4"})
        return batch_response(vehicle_id)
    api = RawAPIClient(tmp_path, {**source_config(), "workers": 4, "min_interval_seconds": .25,
                                 "max_attempts": 1}, transport=transport)
    results = api.fetch_vehicles(batch_requests())
    assert isinstance(results[0][1], APIError)
    assert min(start for vehicle_id, start in starts.items() if vehicle_id != "1") >= first_response_at[0] + .35


def test_batch_worker_interruption_commits_other_completed_responses_before_raising(tmp_path):
    def transport(url, *args):
        vehicle_id = url.rsplit("/", 1)[-1]
        if vehicle_id == "1":
            raise KeyboardInterrupt()
        return batch_response(vehicle_id)
    api = batch_client(tmp_path, transport, FakeClock())
    with pytest.raises(KeyboardInterrupt):
        api.fetch_vehicles(batch_requests())
    assert len(api.entries) == 3
    resumed = batch_client(tmp_path, lambda *args: pytest.fail("Committed responses must resume"), FakeClock())
    results = resumed.fetch_vehicles(batch_requests((2, 3, 4)))
    assert all(error is None for _, error in results) and resumed.request_count == 0


def test_batch_request_budget_is_global_and_preserves_completed_responses(tmp_path):
    api = batch_client(tmp_path, lambda url, *args: batch_response(url.rsplit("/", 1)[-1]),
                       FakeClock(), max_requests=2)
    results = api.fetch_vehicles(batch_requests())
    assert api.request_count == 2 and len(api.entries) == 2
    assert sum(error is None for _, error in results) == 2
    assert sum(isinstance(error, RequestLimitReached) for _, error in results) == 2
    assert len(api.manifest_path.read_text().splitlines()) == 2


def test_batch_individual_schema_failure_is_not_a_successful_raw_record(tmp_path):
    def transport(url, *args):
        vehicle_id = url.rsplit("/", 1)[-1]
        return batch_response(99 if vehicle_id == "2" else vehicle_id)
    api = batch_client(tmp_path, transport, FakeClock())
    results = api.fetch_vehicles(batch_requests())
    assert isinstance(results[1][1], APIError) and results[1][1].category == "schema"
    assert not (tmp_path / "vehicles/2.json").exists()
    assert len(api.entries) == 3


def test_batch_manifest_crash_recovers_journal_without_changing_original_bytes(tmp_path):
    api = batch_client(tmp_path, lambda url, *args: batch_response(url.rsplit("/", 1)[-1]), FakeClock())
    def crash(entry):
        raise OSError("simulated manifest crash")
    api._append_manifest = crash
    with pytest.raises(OSError, match="manifest crash"):
        api.fetch_vehicles(batch_requests())
    original = (tmp_path / "vehicles/1.json").read_bytes()
    assert len(list((tmp_path / "pending_requests").glob("*.json"))) == 1
    resumed = batch_client(tmp_path, lambda *args: pytest.fail("Published response must recover"), FakeClock())
    results = resumed.fetch_vehicles(batch_requests((1,)))
    assert results[0][1] is None and resumed.cache_hits == 1
    assert (tmp_path / "vehicles/1.json").read_bytes() == original


@pytest.mark.parametrize("requests", [batch_requests((1, 1)), batch_requests((1, 2, 3, 4, 5)),
                                       [{"endpoint": "vehicle/menu/year", "file": "menus/year.json"}]])
def test_batch_guards_reject_duplicates_oversized_and_menu_requests(tmp_path, requests):
    api = batch_client(tmp_path, lambda *args: pytest.fail("Invalid batch must not request"), FakeClock())
    with pytest.raises(ValueError):
        api.fetch_vehicles(requests)
    assert api.request_count == 0


def menu_batch_requests(years=(2015, 2016, 2017, 2018)):
    return [{"endpoint": "vehicle/menu/model", "params": {"year": year, "make": "Honda"},
             "file": f"menus/models_{year}.json", "expected": {"menu": "model"}} for year in years]


def test_menu_batch_overlaps_at_global_rate_and_keeps_exact_bytes_single_writer(tmp_path):
    import threading
    import time
    from urllib.parse import parse_qs, urlparse
    lock = threading.Lock()
    active, peak = 0, 0
    starts = []
    bodies = {"2015": b'{ "menuItem": { "text": "Civic", "value": "Civic" } }\n',
              "2016": b'null\n', "2017": b'{"menuItem": []}\n',
              "2018": b'{"menuItem":[{"text":"Fit","value":"Fit"}]}\n'}
    def transport(url, headers, timeout):
        nonlocal active, peak
        assert headers["Accept"] == "application/json"
        year = parse_qs(urlparse(url).query)["year"][0]
        with lock:
            starts.append(time.monotonic())
            active += 1
            peak = max(peak, active)
        time.sleep(.8)
        with lock:
            active -= 1
        return response(bodies[year])
    api = batch_client(tmp_path, transport)
    append = api._append_manifest
    writers = []
    def observed_append(entry):
        writers.append(threading.get_ident())
        append(entry)
    api._append_manifest = observed_append
    results = api.fetch_menus(menu_batch_requests())
    assert 2 <= peak <= 4
    assert all(b - a >= .23 for a, b in zip(starts, starts[1:]))
    assert all(error is None for _, error in results)
    assert results[1] == (None, None)  # JSON null is a successful empty cache entry.
    assert writers == [threading.get_ident()] * 4
    entries = [json.loads(line) for line in api.manifest_path.read_text().splitlines()]
    for request, entry in zip(menu_batch_requests(), entries, strict=True):
        body = bodies[str(request["params"]["year"])]
        assert entry["kind"] == "menu" and entry["expected"] == {"menu": "model"}
        assert entry["params"] == request["params"]
        assert (tmp_path / entry["file"]).read_bytes() == body
        assert entry["sha256"] == hashlib.sha256(body).hexdigest()
    resumed = batch_client(tmp_path, lambda *args: pytest.fail("All menu shapes must resume from cache"))
    assert resumed.fetch_menus(menu_batch_requests()) == results
    assert resumed.request_count == 0 and resumed.cache_hits == 4


def test_menu_batch_retry_after_is_shared_with_later_vehicle_batch(tmp_path):
    clock = FakeClock()
    attempts, starts = [], []
    def transport(url, *args):
        starts.append(clock.seconds)
        attempts.append(url)
        if "/menu/" in url:
            if len(attempts) == 1:
                return response(b"busy", 429, {"Retry-After": "7"})
            return response({"menuItem": None})
        return batch_response(url.rsplit("/", 1)[-1])
    api = batch_client(tmp_path, transport, clock)
    assert api.fetch_menus(menu_batch_requests((2015,)))[0][1] is None
    assert all(error is None for _, error in api.fetch_vehicles(batch_requests()))
    assert starts[0] == 0 and starts[1] >= 7
    assert all(start >= 7.25 for start in starts[2:])
    lines = [json.loads(line) for line in api.manifest_path.read_text().splitlines()]
    assert [entry["attempt"] for entry in lines if entry["kind"] == "menu"] == [1, 2]
    assert lines[0]["retryable"] and (tmp_path / lines[0]["file"]).read_bytes() == b"busy"


def test_menu_batch_budget_preserves_successful_siblings_for_offline_resume(tmp_path):
    api = batch_client(tmp_path, lambda *args: response({"menuItem": None}), FakeClock(), max_requests=2)
    results = api.fetch_menus(menu_batch_requests())
    assert api.request_count == len(api.entries) == 2
    assert sum(error is None for _, error in results) == 2
    assert sum(isinstance(error, RequestLimitReached) for _, error in results) == 2
    completed = [request for request, (_, error) in zip(menu_batch_requests(), results, strict=True) if error is None]
    resumed = batch_client(tmp_path, lambda *args: pytest.fail("Successful siblings must be durable"), FakeClock())
    assert all(error is None for _, error in resumed.fetch_menus(completed))
    assert resumed.request_count == 0


@pytest.mark.parametrize("bad_body", [b'<html>not JSON</html>', b'{"menuItem":{"text":"bad","value":"not-id"}}'])
def test_menu_batch_schema_failure_commits_valid_siblings_without_replacing_failed_menu(tmp_path, bad_body):
    from urllib.parse import parse_qs, urlparse
    requests = [{**item, "endpoint": "vehicle/menu/options", "expected": {"menu": "options"}}
                for item in menu_batch_requests()]
    def transport(url, *args):
        year = parse_qs(urlparse(url).query)["year"][0]
        return response(bad_body if year == "2016" else {"menuItem": {"text": "Automatic", "value": "123"}})
    api = batch_client(tmp_path, transport, FakeClock())
    results = api.fetch_menus(requests)
    assert isinstance(results[1][1], APIError) and results[1][1].category == "schema"
    assert api.request_count == 4 and len(api.entries) == 3
    assert not (tmp_path / requests[1]["file"]).exists()
    failed = next(json.loads(line) for line in api.manifest_path.read_text().splitlines() if not json.loads(line)["valid"])
    assert failed["kind"] == "menu" and (tmp_path / failed["file"]).read_bytes() == bad_body


@pytest.mark.parametrize("requests", [menu_batch_requests((2015, 2015)), menu_batch_requests((2015, 2016, 2017, 2018, 2019)),
    [{"endpoint": "vehicle/123", "file": "menus/wrong.json"}],
    [{"endpoint": "vehicle/menu/not-documented", "file": "menus/wrong.json"}],
    [{"endpoint": "vehicle/menu/model", "file": "menus/wrong.json", "expected": {"menu": "options"}}]])
def test_menu_batch_endpoint_identity_and_distinctness_guards(tmp_path, requests):
    api = batch_client(tmp_path, lambda *args: pytest.fail("Invalid menu batch must not request"), FakeClock())
    with pytest.raises(ValueError):
        api.fetch_menus(requests)
    assert api.request_count == 0


def test_menu_batch_corrupt_cache_is_rejected_without_network(tmp_path):
    api = batch_client(tmp_path, lambda *args: response({"menuItem": None}), FakeClock())
    request = menu_batch_requests((2015,))[0]
    api.fetch_menus([request])
    raw = tmp_path / request["file"]
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(CorruptCacheError, match="Checksum"):
        batch_client(tmp_path, lambda *args: pytest.fail("Corrupt evidence must not be replaced"), FakeClock())
