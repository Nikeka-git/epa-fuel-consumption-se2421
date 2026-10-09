# Independent integration review, 9 October 2026

Reviewed the current `api.py`, `collect.py`, `clean.py`, `features.py`, `split.py`, `train.py` and `evaluate.py`, including collector and modeling offline tests. The ordinary collection → cleaning → split → training path has compatible file and column interfaces. No current target-proxy leakage was found: training selects exactly the seven predictors, every CV pipeline is fitted on its training fold, vehicle families stay disjoint, and model selection is persisted before test prediction.

The following actual issues were identified and repaired during this session:

1. A valid cached vehicle could previously reset an inventory row with conflicting options-menu identities from `failed` to `fetched` on resume. A temporary fake-catalogue reproduction showed two conflicting identities and status `fetched` after resume. The collector agent added a conflicting-identity check that preserves quarantine on resume and a regression test.
2. Resume preparation could previously fail checksum validation after writing `status=running`, leaving an ended operation marked running with no finish time. A temporary fake-catalogue reproduction confirmed this state after `CorruptCacheError`. The collector agent now records `interrupted`, a completion timestamp and preparation failure reason before rethrowing, with an expanded regression test.
3. Correctly quarantining an inventory row did not originally protect cleaning: the cleaner read every raw vehicle file without consulting collector inventory. After explicit parent authorization, the cleaner now validates every available inventory record, requires `status=fetched`, checks its ID and unique menu identity, and verifies year/manufacturer/model against the raw record. Failed, pending, missing and conflicting inventory entries are excluded with separate reasons. Legacy/test snapshots without inventory are permitted and explicitly labelled as such in the summary. Tests verify both conflicts and a mismatched menu identity.
4. The optional cleaner cache-index fallback expected a flat mapping although the collector writes an `entries` wrapper. The cleaner now supports this actual format. Requests JSONL remains the primary provenance source.
5. The cleaner CLI originally interpreted `examples/smoke/raw` as a bare name under `data/raw`. Its `resolve_snapshot` helper now accepts existing paths relative to the current directory or project root, absolute paths, and otherwise names under `data/raw`. A deterministic path-resolution test covers these forms.

Verified preservation rules:

- Source response bytes and historical pilot files were not changed.
- The cleaner rejects changed snapshot/configuration when an existing processed table belongs to another benchmark configuration.
- Positive/finite MPG and converted L/100 km are both required; no target imputation or unrounded-MPG fallback exists.
- Duplicate candidates are created without target and only complete raw equivalence is consolidated. Unknown physical differences remain retained and counted provisionally.
- Collector attempt count, fetched raw records and cleaned rows are distinct report quantities. A bounded run reports partial collection and cannot claim a complete full catalogue.
- `--allow-small` permits a development run without concealing that it fails the 1,000-row requirement. Training and split metadata explicitly label small datasets.
- Saved predictions are checked against raw dataset targets and residual definitions before error tables are regenerated; evaluation does not refit models.

Tests executed during review:

- Entire available suite at the time of invocation: **80 passed in 48.50s**.
- Collector plus cleaner suite after collector repairs (before the final cleaner additions): **50 passed in 12.39s**.
- Final focused cleaner suite after inventory/path additions: **42 passed in 3.43s**.

The full raw-field dictionary still records every observed key, JSON types, missing counts and feature disposition, while full source meanings are explicitly marked unreviewed. This remains documentation work for the complete data card; it should not be described as a finished semantic dictionary. Unknown group aliases, unresolved duplicate candidates, small sample support and limited collection coverage remain explicit limitations rather than hidden claims.

No remaining blocker was found for publishing the initial bounded development snapshot and runnable implementation. A >=1,000-row benchmark, reviewed data card, full inventory/alias review and complete academic deliverables remain future work.
