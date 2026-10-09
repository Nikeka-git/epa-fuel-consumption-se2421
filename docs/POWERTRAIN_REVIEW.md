# Supplemental powertrain review before benchmark freeze

Review date: 2026-10-09. These decisions were fixed before the main train/test split and any main model fit. The original API bytes are preserved. No target value, model error or test score informed these rules.

The API's blank `atvType`, `evMotor` and `fuelType2` fields do not independently prove that a car is non-hybrid. Two immutable test fixtures demonstrate this: US MY2022 Volvo S60 B5 AWD (44187) and XC60 B5 AWD (44205). Manufacturer B-badge definitions and US announcements identify these as mild hybrids. The reviewed Volvo rule is bounded to B4/B5/B6 names and MY2022-2025; it does not remove earlier T-badged vehicles.

Audi of America's MY2021 launch releases and its MY2021-2022 technical bulletin confirm mild-hybrid 2.0L four-cylinder A4/A5/Q5 powertrains. The additional rule uses precisely those years, names and engine conditions. It does not carry evidence backward to MY2020 or remove S4/S5/SQ5/Q3 configurations.

SQ7/SQ8 4.0L V8 configurations present a separate uncertainty. The US parts catalogue uses an MHEV label, while service bulletins are conditional on starter-alternator equipment; another bulletin concerns a chassis 48V system. A battery or chassis system alone cannot establish a hybrid powertrain. Consequently, observed MY2020-2025 SQ7/SQ8 4.0L V8 records are explicitly quarantined as **powertrain uncertain**, not asserted to be confirmed hybrids. This conservative restriction narrows the scope of the resulting benchmark.

All source URLs, findings, model-year boundaries, access limitations and available hashes are recorded in [the executable rule registry](../configs/powertrain_exclusions.json). The cleaning summary embeds the registry and module hashes. Inference requires a model designation for covered manufacturer/year ranges even when the predictor uses structured features only. Compatible but unresolved engine conditions are quarantined in cleaning and rejected at inference. Scope identity does not become an extra Midterm prediction feature.

Additional manufacturer material was reviewed for BMW, Mercedes-Benz, Jaguar/Land Rover and Mazda, but no target-based or European-to-US inference was added. This remains a focused review of observed records rather than an exhaustive independent certification of every retained vehicle. Residual omissions in EPA technology metadata are a limitation of the dataset.

Supplemental rules are target-neutral scope restrictions, not model input features. The final exclusion inventory reports their effects alongside API-labelled hybrid and alternative-fuel exclusions. Start/stop alone, a manufacturer name, battery sentinel values, or a later year's technology label never justify exclusion.
