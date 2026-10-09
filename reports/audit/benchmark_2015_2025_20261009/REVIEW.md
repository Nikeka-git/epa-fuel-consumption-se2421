# Independent benchmark audit: benchmark_2015_2025_20261009

Mode: final_stopped_snapshot. Verified committed records: 4500. Scope-eligible before duplicates: 3247. Immutable integrity issue rows: 0.

- Review model_family_fallbacks.csv and model_family_consistency.csv from source evidence before freeze; do not infer aliases by model-name prefixes.
- No target outlier pruning or target imputation was performed. Counts describe catalogue configurations, not vehicle sales.
- Unknown/non-allowed classes and alternative-technology labels are reported; this audit does not extend the allowlist.
