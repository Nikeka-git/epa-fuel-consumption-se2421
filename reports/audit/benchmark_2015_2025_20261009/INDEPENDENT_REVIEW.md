# Independent pre-freeze review: benchmark_2015_2025_20261009

Decision: **PASS for the observed cleaned dataset; freeze with the specified reviewed family alias map.** Review by a separate Codex agent as part of the AI demonstration. This is not human course approval or an exhaustive powertrain certification. The review did not create a split, fit models, modify raw bytes, or revise historical cleaning metadata.

The stopped bounded collection contains **4,500 raw vehicle records** and **8,568 successful cached responses / manifest attempts**, with no failed attempts. Its status is `partial`; a complete catalogue census is not claimed. Directly rehashing all committed cache files and reconciling manifest URLs, request IDs, response hashes, and vehicle/menu identities found **zero discrepancies**. See [the audit summary](audit_summary.json) and [raw integrity evidence](raw_integrity.csv).

A separate reconstruction from raw records, without calling the cleaner's normalization function, reproduced exactly the **3,247 retained IDs** and **1,253 exclusions**. All seven retained predictors agree with their raw fields, all converted targets equal `235.2145833333333 / comb08`, and no target or input integrity discrepancy was found. Targets remain **6.0311–26.1350 L/100 km**; no outlier pruning, target imputation, or target-driven scope decision was performed.

## Categories and scope

The observed class inventory supports the predeclared allowlist for this snapshot. The retained data contain **12 classes**: six passenger-car classes, two station-wagon classes, and four Small/Standard SUV 2WD/4WD classes. `Large Station Wagons` is allowlisted but unobserved. Nine observed out-of-scope classes are consistently excluded: two minivan classes, four pickup classes, two Special Purpose Vehicle classes, and Passenger Vans. The allowlist was not expanded. The historical `class_allowlist_status` remains unchanged; this document records the actual completed review.

Retained primary fuels are **1,934 Premium Gasoline**, **1,283 Regular Gasoline**, and **30 Midgrade Gasoline**. Every retained `fuelType2`, `atvType`, and `evMotor` is an empty source string, and every retained `phevBlended` is the source string `false`. Required powertrain keys are present and non-null in all 4,500 raw records; no unknown PHEV boolean was observed. Empty strings mean the source supplies no label, not independently proven absence of hybrid equipment.

The nonempty raw technology inventory is Plug-in Hybrid (121), EV (388), Hybrid (292), FCV (8), FFV (103), Diesel (70), eFCV (1), and Bifuel (CNG) (1). All are excluded by the predeclared nonempty-technology policy; a rare or unconfirmed spelling cannot enter the gasoline-only benchmark. Nonempty motor strings, including battery-like labels and `TBD`, are retained in raw evidence and excluded, rather than interpreted numerically or certified as specific technology. See [category counts](category_counts.csv) and [scope decisions](raw_scope_decisions.csv).

## Supplemental powertrain rules

Independent reason counts overlap and must not be added as unique exclusions.

| Rule | Matching raw records | Additional exclusions beyond ordinary source filters |
| --- | ---: | ---: |
| Volvo B-badge mild hybrids, bounded MY2022–2025 | 24 | 3 |
| Audi A4/A5/Q5 2.0L four-cylinder MY2021–2022 | 4 | 0 |
| Audi SQ7/SQ8 4.0L V8 uncertainty quarantine | 3 | 3 |

The three additional Volvo records are **44187, 44205, and 44481**; their technology/motor/secondary-fuel labels are blank. The three SQ7/SQ8 records are **47806, 43029, and 47178**. They are explicitly excluded as **powertrain uncertain**, not declared proven hybrids. The four Audi A4/A5 matches were already excluded by Hybrid/motor labels. These rules therefore remove **six additional unique records**, not 31. Their model-year/name/engine conditions are target-neutral and fixed before main fitting.

The [rule registry](../../../configs/powertrain_exclusions.json) and [powertrain review](../../../docs/POWERTRAIN_REVIEW.md) preserve primary-source findings, exact bounds, and access limitations. This independent review rechecked rule application to raw identities; it did not newly fetch every external source. Blank metadata outside these focused corrections remains a limitation, so the dataset must not be described as independently certified non-hybrid in every case.

## Distinct configurations, families, and missing values

Independent target-free technical candidate formation found **zero duplicate groups**. The cleaner and audit likewise report zero removed confirmed duplicates and zero unresolved candidate groups. The distinct count is therefore **3,247 source configurations** under the documented conservative rule, rather than merely 3,247 unique IDs or seven-feature combinations.

No retained record needs a model-name fallback for missing `baseModel`. Eighteen identical full model names nevertheless occur under multiple source baseModel labels. The separate [reviewed alias map](family_review/final/reviewed_family_aliases.csv) was checked against an independently reconstructed identity-only graph: **18 literal edges, 13 connected components, 86 explicit aliases**. The complete propagated map matches that graph; it changes 112 row groups and reduces 405 source groups to 389. After mapping, no identical manufacturer/full-model name remains in multiple groups.

Freeze must use that exact map, SHA-256 `9dc5e6f72b7b7349642946730a43236e921ff470b9bb78297d6afa1d24f6c559`. Broad BMW M, Audi RS, and MINI John Cooper Works connections are conservative source-taxonomy grouping; physical-generation equivalence is not asserted. No prefix aliases, target values, or model errors were used. Further source relations may remain undiscovered.

All **seven structured predictors have zero missing values** in this snapshot. Original engine description is missing in **777 / 3,247 rows**; model names are present. Preserve these rows for the later paired text comparison and use the declared empty-text behavior. The sample spans **all eleven model years 2015–2025, 49 manufacturers, and 12 classes**, with unequal per-year counts; it is not sales weighted.

The full dictionary covers **95 observed raw fields**. Seven encodings remain explicitly unconfirmed: `battery`, `cylDeact`, `cylDeactYesNo`, `mpgRevised`, `range`, `rangeCity`, and `rangeHwy`. All seven are audit-only or leakage/output exclusions and do not enter any predictor set. The seven structured predictors have documented meanings. Unconfirmed encodings must not be described as completely verified source semantics.

Machine-readable decisions and input hashes are in [review_decision.json](review_decision.json). No historical `cleaned_summary.json`, source/config bytes, table, or split was altered by this review.
