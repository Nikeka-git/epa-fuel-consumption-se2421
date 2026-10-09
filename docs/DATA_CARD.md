# Карточка основного benchmark

**3247 различных пригодных конфигураций**; группа SE-2421. Команда проекта: Tsybus Nikita и Bakytzhan Kassymgali. Источник: [FuelEconomy.gov API](https://www.fueleconomy.gov/feg/ws/index.shtml). Единица строки — EPA catalogue configuration определенного модельного года с уникальным vehicle ID, не проданный экземпляр, VIN или владелец.

## Сбор и происхождение

Snapshot: `benchmark_2015_2025_20261009`. Собственный menu traversal → individual vehicle request, JSON Accept header. Получено 4500 vehicle records и 8568 сохраненных HTTP responses/attempts; failures 0. Начало UTC `2026-10-09T00:55:02.056715+00:00`, окончание `2026-10-09T02:14:20.303923+00:00`; timezone Asia/Qyzylorda.

Сбор чередовал все годы 2015–2025 и марки из меню API с seed 42. Первоначальный бюджет 6 000 исходных записей сокращен до 4 500 по числу пригодных записей промежуточного аудита, до обучения моделей. Цель 3 000 достигнута. [Provenance](../evidence/benchmark_collection_provenance.json) фиксирует это решение, контролируемые перезапуски и версии кода. Config сохранен; поле status в нем — историческая метка до запуска, фактические статусы находятся в artifacts.

Статус snapshot — **partial**, причина остановки — заданный лимит. Enumeration complete и full catalogue complete — false. Есть обнаруженные, но не загруженные IDs; отсутствие зарегистрированных ошибок не означает полный обход. Вероятности включения не равны; количества конфигураций не являются продажами. Запросы, прерванные при контролируемых остановках, могли не оставить ответа; counts относятся к сохраненному manifest.

Исходные ответы и меню, журнал запросов, URL, HTTP status, timestamps и SHA-256 сохранены в `data/raw/benchmark_2015_2025_20261009/`. [Окончательный audit](../reports/audit/benchmark_2015_2025_20261009/audit_summary.json) сверил 4 500 исходных ответов и 3 247 очищенных строк: проблем целостности, кандидатов в дубликаты и fallback для отсутствующего baseModel — 0.

## Область и очистка

US gasoline cars, station wagons и SUVs. Исключены hybrids/MHEV/PHEV, electric/alternative/secondary fuels, trucks/vans/minivans/special-purpose classes и недопустимые metadata/target/engine fields. Пустые отдельные допустимые predictors остаются для train-fitted imputation; target не импутируется и не заменяется `comb08U`.

Последовательные counts (не путать с пересекающимися independent reasons):

| stage | before | removed | after |
| --- | --- | --- | --- |
| raw_integrity | 4500 | 0 | 4500 |
| required_identity | 4500 | 0 | 4500 |
| model_year | 4500 | 0 | 4500 |
| powertrain_metadata | 4500 | 0 | 4500 |
| primary_fuel | 4500 | 467 | 4033 |
| secondary_fuel | 4033 | 225 | 3808 |
| alternative_technology | 3808 | 292 | 3516 |
| electric_motor | 3516 | 0 | 3516 |
| phev | 3516 | 0 | 3516 |
| hybrid_text | 3516 | 0 | 3516 |
| reviewed_powertrain | 3516 | 6 | 3510 |
| vehicle_class | 3510 | 263 | 3247 |
| target | 3247 | 0 | 3247 |
| engine_specs | 3247 | 0 | 3247 |
| confirmed_complete_source_duplicates | 3247 | 0 | 3247 |

Дополнительные правила удалили 6 записей: 3 подтвержденных Volvo MHEV с пропущенными EPA-метками и 3 Audi SQ7/SQ8 с неопределенной технологией. Полные независимые совпадения правил пересекаются с API hybrid exclusions. [Review](POWERTRAIN_REVIEW.md) и [registry](../configs/powertrain_exclusions.json) показывают условия марки, года, названия и двигателя, а также ограничения источников. Правила заданы без использования величины target до split. Source metadata остается потенциально неполным.

Разные IDs с одинаковыми семью X не удаляются как дубли. Формирование кандидатов не использует target; удаляются только полностью совпадающие source configurations. Подтвержденных удалений и unresolved groups — 0. Выбросы target не удалялись.

## Поля и утечки

Target: `target_l100km = 235.2145833333333 / comb08`, US MPG→L/100 km. `vehicle_id` — provenance key. Числовые X: model_year/displacement_l/cylinders. Категориальные X: manufacturer/transmission/drivetrain/vehicle_class. Final text: original model_name и доступный engine_description. base_model/model_group — только grouping/audit. Остальные fuel-economy/cost/emissions/score/id/date/source fields исключены из inputs по умолчанию.

Для 95 исходных полей подготовлены [машинный словарь](../data/interim/raw_field_dictionary.json) и [CSV](../data/interim/raw_field_dictionary.csv). 86 опираются на официальный API guide; 2 дополнительно описывают структуру и наблюдаемый тип. Значения кодировок семи полей battery/cylDeact/cylDeactYesNo/mpgRevised/range/rangeCity/rangeHwy не подтверждены; это отмечено в словаре. Они не являются входами моделей. Полная семантическая сертификация всех source encodings не заявляется. [Schema](../data/processed/schema.json) и [контракт](DATA_CONTRACT.md) описывают внутренние поля.

Пропуски в сохраненной таблице:

| field | missing |
| --- | --- |
| model_year | 0 |
| displacement_l | 0 |
| cylinders | 0 |
| manufacturer | 0 |
| transmission | 0 |
| drivetrain | 0 |
| vehicle_class | 0 |
| model_name | 0 |
| engine_description | 777 |

Пустой engine text не генерируется и не исключает строку. Final sanitizer удаляет GUZZLER и явные economy/cost/emissions/score markers; vocabulary fit только в training fold.

## Покрытие

Производителей: **49**. Count по годам:

| model_year | configurations |
| --- | --- |
| 2015 | 310 |
| 2016 | 322 |
| 2017 | 330 |
| 2018 | 332 |
| 2019 | 325 |
| 2020 | 325 |
| 2021 | 317 |
| 2022 | 289 |
| 2023 | 276 |
| 2024 | 227 |
| 2025 | 194 |

Count по классам:

| vehicle_class | configurations |
| --- | --- |
| Compact Cars | 390 |
| Large Cars | 308 |
| Midsize Cars | 474 |
| Midsize Station Wagons | 30 |
| Minicompact Cars | 190 |
| Small Sport Utility Vehicle 2WD | 267 |
| Small Sport Utility Vehicle 4WD | 496 |
| Small Station Wagons | 108 |
| Standard Sport Utility Vehicle 2WD | 111 |
| Standard Sport Utility Vehicle 4WD | 294 |
| Subcompact Cars | 232 |
| Two Seaters | 347 |

Полные категории и missingness доступны в interim/audit. Содержательная EDA использует train target.

## Freeze и ограничения

2 628 train / 619 test; 311 / 78 семейств; test — 20% source groups, seed 42. На train — пять GroupKFold folds. Manufacturer+baseModel через годы дополнен reviewed literal-name alias map; все решения и identity evidence сохранены. Пересечения ID/groups/CV — 0. Это не прогноз будущих модельных лет и не доказательство независимости всех платформ или поколений.

Dataset SHA-256: `f9152b6d24577b2abeb5c149fd5a247a49d9c098c7d0dd50202ed8d1ad652739`. Split SHA-256: `8ad5db3c3f606ef862fdbf8c72c215da93a8ec786ac208351370f8180352b9df`. Config/rule/cleaner/module/model hashes сохраняются в отдельных metadata. Все три этапа используют эти же строки и test; test после Midterm является повторно используемым сравнительным benchmark, а не новым скрытым holdout.

EPA estimate отличается от фактического дорожного расхода; rounded MPG ограничивает точность target. Начальные features не включают массу, мощность и аэродинамику. Редкие подгруппы имеют малую поддержку; результаты не sales-weighted. [RESULTS](../reports/project_v1/RESULTS.md) показывает actual errors и ограничения.

## Доступ и условия

Реальные requests подтверждают отсутствие необходимости login; данные описывают объекты, не частных людей. В `evidence/source_policy/` сохранены official API documentation и [ORNL Privacy/Security policy](https://www.fueleconomy.gov/feg/ORNL-disclaimer.htm) с датой и hashes. Документы допускают образовательное некоммерческое использование. Пользователь сообщил о проверке robots.txt; собственная попытка получить его завершилась закрытием соединения, поэтому сохраненная успешная robots verification не заявляется. Изображения и персональные driver MPG records не собирались.

Исторический smoke из 250 записей имеет отдельные config/split/results в `examples/smoke/`; он не смешан с основным benchmark.
