"""Small simulated catalogues verify resume, coverage, and provenance."""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from fuel_consumption.api import APIError, CorruptCacheError, RawAPIClient
from fuel_consumption.collect import Collector, SnapshotLock
from test_api import FakeClock, response, source_config


class FakeCatalogue:
    def __init__(self):
        self.calls = []
        self.records = {}
        self.fail_vehicle = None
        self.wrong_identity = None
        for year in (2015, 2025):
            for make_index, (make, models) in enumerate((("Honda", ["Civic", "Fit"]), ("Toyota", ["Corolla", "RAV4"]))):
                for model_index, model in enumerate(models):
                    for option in (1, 2):
                        vehicle_id = str(year * 100 + make_index * 10 + model_index * 2 + option)
                        self.records[vehicle_id] = {"id": vehicle_id, "year": str(year), "make": make,
                                                    "model": model, "comb08": "30", "VClass": "Compact Cars"}

    @staticmethod
    def menu(values):
        items = [{"text": value, "value": value} for value in values]
        return {"menuItem": items[0] if len(items) == 1 else items}

    def __call__(self, url, headers, timeout):
        self.calls.append(url)
        route = urlparse(url).path.rsplit("/", 1)[-1]
        params = {key: values[0] for key, values in parse_qs(urlparse(url).query).items()}
        if route == "year":
            return response(self.menu(["2015", "2025"]))
        if route == "make":
            return response(self.menu(["Honda", "Toyota"]))
        if route == "model":
            return response(self.menu(["Civic", "Fit"] if params["make"] == "Honda" else ["Corolla", "RAV4"]))
        if route == "options":
            matches = [key for key, row in self.records.items()
                       if (row["year"], row["make"], row["model"]) == (params["year"], params["make"], params["model"])]
            return response(self.menu(matches))
        if route == self.fail_vehicle:
            return response(b"missing", status=404)
        row = dict(self.records[route])
        if route == self.wrong_identity:
            row["model"] = "Wrong model"
        return response(row)


def silent(*args, **kwargs):
    pass


def make_collector(tmp_path, catalogue, **kwargs):
    clock = FakeClock()
    def factory(path, config, **client_kwargs):
        return RawAPIClient(path, config, transport=catalogue,
                            monotonic=clock.monotonic, sleep=clock.sleep,
                            now=clock.now, **client_kwargs)
    workers = kwargs.pop("workers", 1)
    config = {"source": {**source_config(), "years": [2015, 2025], "workers": workers,
                         "min_interval_seconds": 1 / workers}}
    return Collector(tmp_path, config, "test_snapshot", client_factory=factory, progress=silent, **kwargs)


def test_bounded_smoke_rotates_years_and_manufacturers(tmp_path):
    catalogue = FakeCatalogue()
    collector = make_collector(tmp_path, catalogue, max_vehicles=4, seed=42)
    metadata = collector.run()
    records = [json.loads(path.read_text()) for path in (collector.path / "vehicles").glob("*.json")]
    assert {(int(row["year"]), row["make"]) for row in records} == {
        (2015, "Honda"), (2015, "Toyota"), (2025, "Honda"), (2025, "Toyota")}
    assert len(records) == 4 and metadata["fetched_vehicle_records"] == 4
    assert metadata["full_catalogue_complete"] is False and metadata["status"] == "partial"
    assert metadata["collection_mode"] == "interleaved_bounded"
    assert (collector.path / "inventory.csv").is_file()
    assert (collector.path / "inventory.json").is_file()


def test_same_completed_bounded_resume_does_not_request_cache(tmp_path):
    catalogue = FakeCatalogue()
    first = make_collector(tmp_path, catalogue, max_vehicles=4, seed=42)
    first.run()
    calls_before = len(catalogue.calls)
    before = {path.name: path.read_bytes() for path in (first.path / "vehicles").glob("*.json")}
    resumed = make_collector(tmp_path, catalogue, resume=True, max_vehicles=4, seed=42)
    result = resumed.run()
    assert len(catalogue.calls) == calls_before
    assert resumed.client.request_count == 0
    assert result["fetched_vehicle_records"] == 4
    assert {path.name: path.read_bytes() for path in (first.path / "vehicles").glob("*.json")} == before


def test_request_budget_pause_resumes_discovered_unfetched_ids(tmp_path):
    catalogue = FakeCatalogue()
    first = make_collector(tmp_path, catalogue, max_requests=6, max_vehicles=8, seed=42)
    first.run()
    assert first.client.request_count == 6
    assert len(first.inventory) == 2 and first._fetched_count() == 1
    first_vehicle_url = next(url for url in catalogue.calls if url.rsplit("/", 1)[-1].isdigit())
    resumed = make_collector(tmp_path, catalogue, resume=True, max_requests=100, max_vehicles=8, seed=42)
    result = resumed.run()
    assert result["fetched_vehicle_records"] == 8
    assert catalogue.calls.count(first_vehicle_url) == 1
    manifest = [json.loads(line) for line in (first.path / "requests.jsonl").read_text().splitlines()]
    success_files = [row["file"] for row in manifest if row["valid"]]
    assert len(success_files) == len(set(success_files))


def test_unbounded_catalogue_enumerates_all_then_fetches_unique_ids(tmp_path):
    catalogue = FakeCatalogue()
    collector = make_collector(tmp_path, catalogue)
    metadata = collector.run()
    assert metadata["status"] == "complete"
    assert metadata["enumeration_complete"] is True
    assert metadata["full_catalogue_complete"] is True
    assert metadata["fetched_vehicle_records"] == 16
    routes = [urlparse(url).path.rsplit("/", 1)[-1] for url in catalogue.calls]
    first_vehicle = next(index for index, route in enumerate(routes) if route.isdigit())
    last_options = max(index for index, route in enumerate(routes) if route == "options")
    assert first_vehicle > last_options
    assert len({route for route in routes if route.isdigit()}) == 16


def test_full_resume_uses_same_cache_without_network_requests(tmp_path):
    catalogue = FakeCatalogue()
    first = make_collector(tmp_path, catalogue)
    first.run()
    calls = len(catalogue.calls)
    resumed = make_collector(tmp_path, catalogue, resume=True)
    result = resumed.run()
    assert len(catalogue.calls) == calls and resumed.client.request_count == 0
    assert result["status"] == "complete" and result["full_catalogue_complete"]


def test_failed_records_are_logged_separately_and_retry_on_resume(tmp_path):
    catalogue = FakeCatalogue()
    catalogue.fail_vehicle = "201501"
    first = make_collector(tmp_path, catalogue)
    metadata = first.run()
    assert metadata["status"] == "partial"
    assert metadata["failed_vehicle_records"] == 1
    assert first.inventory["201501"]["status"] == "failed"
    failures = [json.loads(line) for line in (first.path / "collection_failures.jsonl").read_text().splitlines()]
    assert any(row["kind"] == "vehicle" and row["http_status"] == 404 for row in failures)
    assert not (first.path / "vehicles/201501.json").exists()
    catalogue.fail_vehicle = None
    resumed = make_collector(tmp_path, catalogue, resume=True)
    result = resumed.run()
    assert result["fetched_vehicle_records"] == 16 and result["status"] == "complete"
    assert catalogue.calls.count(source_config()["base_url"] + "vehicle/201501") == 2


def test_wrong_record_identity_does_not_enter_successful_cache(tmp_path):
    catalogue = FakeCatalogue()
    catalogue.wrong_identity = "201501"
    collector = make_collector(tmp_path, catalogue)
    result = collector.run()
    assert result["failed_vehicle_records"] == 1
    assert "disagrees" in collector.inventory["201501"]["error"]
    assert not (collector.path / "vehicles/201501.json").exists()


def test_resume_refuses_corrupt_immutable_vehicle_bytes(tmp_path):
    catalogue = FakeCatalogue()
    first = make_collector(tmp_path, catalogue, max_vehicles=4, seed=42)
    first.run()
    raw_path = next((first.path / "vehicles").glob("*.json"))
    original = raw_path.read_bytes()
    raw_path.write_bytes(original + b" ")
    calls = len(catalogue.calls)
    resumed = make_collector(tmp_path, catalogue, resume=True, max_vehicles=4, seed=42)
    with pytest.raises(CorruptCacheError):
        resumed.run()
    assert len(catalogue.calls) == calls
    assert raw_path.read_bytes() == original + b" "
    assert not (first.path / ".collector.lock").exists()
    metadata = json.loads((first.path / "snapshot.json").read_text())
    assert metadata["status"] == "interrupted"
    assert "Preparation failed" in metadata["stop_reason"]
    assert metadata["finished_at_utc"] is not None


def test_resume_preserves_conflicting_provenance_even_with_valid_cache(tmp_path):
    catalogue = FakeCatalogue()
    first = make_collector(tmp_path, catalogue, max_vehicles=4, seed=42)
    first.run()
    inventory_path = first.path / "inventory.json"
    inventory = json.loads(inventory_path.read_text())
    vehicle_id = next(vehicle_id for vehicle_id, row in inventory["records"].items() if row["status"] == "fetched")
    row = inventory["records"][vehicle_id]
    conflict = dict(row["provenance"][0])
    conflict["model_name"] = "Conflicting model"
    row["provenance"].append(conflict)
    row.update(status="failed", error="Conflicting options-menu identity provenance")
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    resumed = make_collector(tmp_path, catalogue, resume=True, max_vehicles=4, seed=42)
    result = resumed.run()
    assert resumed.inventory[vehicle_id]["status"] == "failed"
    assert resumed.inventory[vehicle_id]["error"] == "Conflicting options-menu identity provenance"
    assert result["full_catalogue_complete"] is False


def test_resume_verifies_completed_options_menu_checksums(tmp_path):
    catalogue = FakeCatalogue()
    first = make_collector(tmp_path, catalogue)
    first.run()
    options_path = next((first.path / "menus").glob("options_*.json"))
    body = options_path.read_bytes()
    options_path.write_bytes(body + b" ")
    calls = len(catalogue.calls)
    with pytest.raises(CorruptCacheError, match="Checksum"):
        make_collector(tmp_path, catalogue, resume=True).run()
    assert len(catalogue.calls) == calls
    assert options_path.read_bytes() == body + b" "


def test_keyboard_interrupt_preserves_inventory_and_resumes(tmp_path):
    catalogue = FakeCatalogue()
    class InterruptingCatalogue:
        interrupted = False
        def __call__(self, url, headers, timeout):
            if urlparse(url).path.rsplit("/", 1)[-1].isdigit() and not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt()
            return catalogue(url, headers, timeout)
    transport = InterruptingCatalogue()
    first = make_collector(tmp_path, transport, max_vehicles=4, seed=42)
    with pytest.raises(KeyboardInterrupt):
        first.run()
    assert first.inventory and first.metadata["status"] == "interrupted"
    assert not (first.path / ".collector.lock").exists()
    resumed = make_collector(tmp_path, catalogue, resume=True, max_vehicles=4, seed=42)
    metadata = resumed.run()
    assert metadata["fetched_vehicle_records"] == 4
    vehicle_rows = [json.loads(line) for line in (first.path / "requests.jsonl").read_text().splitlines()
                    if json.loads(line)["kind"] == "vehicle" and json.loads(line)["valid"]]
    assert len({row["file"] for row in vehicle_rows}) == len(vehicle_rows) == 4


def test_new_snapshot_name_and_resume_scope_guards(tmp_path):
    catalogue = FakeCatalogue()
    first = make_collector(tmp_path, catalogue, max_vehicles=4, seed=42)
    first.run()
    with pytest.raises(ValueError, match="already exists"):
        make_collector(tmp_path, catalogue, max_vehicles=4, seed=42).run()
    with pytest.raises(ValueError, match="scope/seed"):
        make_collector(tmp_path, catalogue, resume=True, max_vehicles=4, seed=41).run()
    with pytest.raises(ValueError, match="simple directory"):
        Collector(tmp_path, {"source": {**source_config(), "years": [2015, 2025]}}, "../escape")


def test_model_cap_partial_status_and_existing_lock(tmp_path):
    catalogue = FakeCatalogue()
    collector = make_collector(tmp_path, catalogue, max_models_per_make=1, seed=42)
    metadata = collector.run()
    assert metadata["status"] == "partial" and metadata["full_catalogue_complete"] is False
    assert metadata["fetched_vehicle_records"] == 8
    assert metadata["completed_model_menus"] == 4
    with SnapshotLock(collector.path):
        with pytest.raises(RuntimeError, match="lock"):
            with SnapshotLock(collector.path, resume=True):
                pass


def test_batch_bounded_collection_keeps_round_robin_scope_and_exact_record_cap(tmp_path):
    catalogue = FakeCatalogue()
    collector = make_collector(tmp_path, catalogue, max_vehicles=5, seed=42, workers=4)
    metadata = collector.run()
    records = [json.loads(path.read_text()) for path in (collector.path / "vehicles").glob("*.json")]
    assert len(records) == metadata["fetched_vehicle_records"] == 5
    assert {(int(row["year"]), row["make"]) for row in records} == {
        (2015, "Honda"), (2015, "Toyota"), (2025, "Honda"), (2025, "Toyota")}
    assert metadata["full_catalogue_complete"] is False
    calls = len(catalogue.calls)
    resumed = make_collector(tmp_path, catalogue, resume=True, max_vehicles=5, seed=42, workers=4)
    result = resumed.run()
    assert result["fetched_vehicle_records"] == 5 and len(catalogue.calls) == calls


def test_batch_request_budget_reports_success_before_pause_and_resume(tmp_path):
    catalogue = FakeCatalogue()
    first = make_collector(tmp_path, catalogue, max_requests=12, max_vehicles=8, seed=42, workers=4)
    result = first.run()
    assert first.client.request_count == 12
    # 11 menu requests discover the first round; only one network slot remains.
    assert result["fetched_vehicle_records"] == 1
    first_urls = {url for url in catalogue.calls if url.rsplit("/", 1)[-1].isdigit()}
    resumed = make_collector(tmp_path, catalogue, resume=True, max_requests=100, max_vehicles=8, seed=42, workers=4)
    result = resumed.run()
    assert result["fetched_vehicle_records"] == 8
    assert all(catalogue.calls.count(url) == 1 for url in first_urls)


def test_batch_unbounded_collection_schema_failure_and_resumed_record(tmp_path):
    catalogue = FakeCatalogue()
    catalogue.wrong_identity = "201501"
    first = make_collector(tmp_path, catalogue, workers=4)
    metadata = first.run()
    assert metadata["failed_vehicle_records"] == 1 and metadata["fetched_vehicle_records"] == 15
    assert first.inventory["201501"]["status"] == "failed"
    catalogue.wrong_identity = None
    resumed = make_collector(tmp_path, catalogue, resume=True, workers=4)
    metadata = resumed.run()
    assert metadata["status"] == "complete" and metadata["fetched_vehicle_records"] == 16


def successful_vehicle_order(snapshot):
    return [json.loads(line)["expected"]["vehicle_id"]
            for line in (snapshot / "requests.jsonl").read_text().splitlines()
            if json.loads(line)["kind"] == "vehicle" and json.loads(line)["valid"]]


@pytest.mark.parametrize("cap", [1, 3, 4, 5, 8, 11, 15])
def test_menu_prefetch_preserves_exact_uninterrupted_seeded_vehicle_prefix(tmp_path, cap):
    sequential = make_collector(tmp_path / 'sequential', FakeCatalogue(), max_vehicles=cap, seed=42, workers=1)
    parallel = make_collector(tmp_path / 'parallel', FakeCatalogue(), max_vehicles=cap, seed=42, workers=4)
    sequential.run()
    parallel.run()
    assert successful_vehicle_order(parallel.path) == successful_vehicle_order(sequential.path)
    assert len(successful_vehicle_order(parallel.path)) == cap


def test_prefetched_models_stay_out_of_logical_state_until_traversal_and_resume_from_cache(tmp_path):
    catalogue = FakeCatalogue()
    first = make_collector(tmp_path, catalogue, max_requests=7, max_vehicles=8, seed=42, workers=4)
    first.run()
    assert first.client.request_count == 7  # year, two makes, four prefetched model menus.
    assert first.state['models_by_pair'] == {} and first.state['completed_models'] == []
    assert first.inventory == {}
    model_urls = [url for url in catalogue.calls if urlparse(url).path.endswith('/model')]
    assert len(model_urls) == 4
    resumed = make_collector(tmp_path, catalogue, resume=True, max_requests=100, max_vehicles=8, seed=42, workers=4)
    result = resumed.run()
    assert result['fetched_vehicle_records'] == 8
    assert all(catalogue.calls.count(url) == 1 for url in model_urls)
    bodies = {path.as_posix(): path.read_bytes() for path in (first.path / 'menus').glob('*.json')}
    calls = len(catalogue.calls)
    done = make_collector(tmp_path, catalogue, resume=True, max_vehicles=8, seed=42, workers=4)
    done.run()
    assert done.client.request_count == 0 and len(catalogue.calls) == calls
    assert {path.as_posix(): path.read_bytes() for path in (first.path / 'menus').glob('*.json')} == bodies


def test_resume_validates_unused_prefetched_model_evidence_without_network(tmp_path):
    catalogue = FakeCatalogue()
    first = make_collector(tmp_path, catalogue, max_requests=7, max_vehicles=8, seed=42, workers=4)
    first.run()
    path = next((first.path / 'menus').glob('model_*.json'))
    body = path.read_bytes()
    path.write_bytes(body + b' ')
    calls = len(catalogue.calls)
    with pytest.raises(CorruptCacheError, match='Checksum'):
        make_collector(tmp_path, catalogue, resume=True, max_vehicles=8, seed=42, workers=4).run()
    assert len(catalogue.calls) == calls and path.read_bytes() == body + b' '


@pytest.mark.parametrize('menu_name,status,expected_attempts', [('model', 404, 1), ('options', 503, 5)])
def test_menu_prefetch_fatal_failure_commits_siblings_and_never_retries_through_traversal(tmp_path, menu_name, status, expected_attempts):
    catalogue = FakeCatalogue()
    failed_urls = []
    def transport(url, *args):
        parsed = urlparse(url)
        params = {key: values[0] for key, values in parse_qs(parsed.query).items()}
        if parsed.path.endswith('/' + menu_name) and params.get('year') == '2015' and params.get('make') == 'Honda':
            failed_urls.append(url)
            return response(b'unavailable', status=status)
        return catalogue(url, *args)
    first = make_collector(tmp_path, transport, max_vehicles=8, seed=42, workers=4)
    with pytest.raises(APIError):
        first.run()
    assert len(failed_urls) == expected_attempts
    assert first.metadata['status'] == 'interrupted' and first._fetched_count() == 0
    assert first.inventory == {} and first.state['completed_models'] == []
    sibling_entries = [entry for entry in first.client.entries.values()
                       if entry['kind'] == 'menu' and entry['expected']['menu'] == menu_name]
    assert len(sibling_entries) == 3
    for entry in sibling_entries:
        assert (first.path / entry['file']).exists()
    resumed = make_collector(tmp_path, catalogue, resume=True, max_vehicles=8, seed=42, workers=4)
    assert resumed.run()['fetched_vehicle_records'] == 8
    assert all(catalogue.calls.count(entry['url']) == 1 for entry in sibling_entries)


def test_menu_prefetch_cached_null_menu_remains_empty_on_resume(tmp_path):
    catalogue = FakeCatalogue()
    def transport(url, *args):
        parsed = urlparse(url)
        params = {key: values[0] for key, values in parse_qs(parsed.query).items()}
        if parsed.path.endswith('/model') and params.get('year') == '2015' and params.get('make') == 'Honda':
            catalogue.calls.append(url)
            return response(None)
        return catalogue(url, *args)
    first = make_collector(tmp_path, transport, max_vehicles=5, seed=42, workers=4)
    first.run()
    null_url = next(url for url in catalogue.calls if urlparse(url).path.endswith('/model')
                    and parse_qs(urlparse(url).query)['year'] == ['2015']
                    and parse_qs(urlparse(url).query)['make'] == ['Honda'])
    resumed = make_collector(tmp_path, transport, resume=True, max_vehicles=9, seed=42, workers=4)
    assert resumed.run()['fetched_vehicle_records'] == 9
    assert catalogue.calls.count(null_url) == 1
    assert all(row['provenance'][0]['manufacturer'] != 'Honda' or row['provenance'][0]['model_year'] != 2015
               for row in resumed.inventory.values())
