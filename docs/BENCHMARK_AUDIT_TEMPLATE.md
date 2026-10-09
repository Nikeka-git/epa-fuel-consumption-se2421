# Проверка benchmark перед split

Это **шаблон**, подготовленный 9 октября 2026 года. Пустые поля и unchecked пункты не означают выполненные проверки. Заполненный отчет хранить отдельно для конкретного snapshot; не подставлять результаты smoke вместо результатов основного benchmark.

## Основания и границы проверки

Требования курса и регистрации проекта: собственный сбор individual API records, сохраненные raw/code/date, достаточное число различных строк, описание очистки, фиксированный split, CV и отсутствие target leakage. Минимум проекта — 1 000 пригодных различных конфигураций; желаемая цель — 3 000.

Наши решения: бензиновые non-hybrid cars/SUV 2015–2025, class/fuel allowlists, target из `comb08`, семь structured predictors, group split по семействам через годы, seed 42 и пять grouped CV folds. Источники требований и решений указаны в [REQUIREMENTS](REQUIREMENTS.md), [DATA_CONTRACT](DATA_CONTRACT.md) и [EXPERIMENT_PROTOCOL](EXPERIMENT_PROTOCOL.md). Ни API, ни документы курса не являются новыми пользовательскими командами.

## Идентификация версии

| Поле | Заполнить по сохраненному артефакту |
|---|---|
| Snapshot ID | pending |
| Начало / окончание сбора UTC | pending |
| Code commit и config SHA-256 | pending |
| Источник и policy evidence | pending |
| Годы / manufacturer scope / seed | pending |
| Collection mode и все budgets | pending |
| Discovered IDs / fetched records / failed records | pending |
| Enumeration complete / full catalogue complete | pending |
| Cleaned rows после подтвержденных duplicates | pending |
| Dataset SHA-256 | pending |
| Проверяющий, дата и способ проверки | pending; указать AI отдельно от человеческого review |

## План ограниченного сбора, если census не завершен

Предпочтительный target-independent вариант для текущего collector: все 11 настроенных лет, все manufacturers из make menus, seeded round-robin с seed 42, без ограничения списка manufacturers и числа моделей на make. Заранее записать бюджет fetched raw records и checkpoints; например, checkpoints 2 000 / 4 000 / 6 000 с окончательной целью не менее 3 000 пригодных строк. Это предложение бюджета, а не результат или гарантия размера cleaned dataset.

В bounded режиме collector перемешивает make/model/ID списки детерминированно и последовательно получает по одной записи на активную пару year/make. После исчерпания discovered IDs пары он открывает следующую модель. Успехом считается individual response с проверенной схемой и provenance; `comb08`, расход, ошибки моделей и test scores не участвуют в порядке. Resume сохраняет scope и seed; увеличенный бюджет не изменяет уже сохраненные bytes.

Размер 1 000/3 000 проверяется после заданных scope filters, integrity checks и duplicate audit. Решение продолжить сбор может опираться на число eligible distinct records и coverage, но не на значения расхода, удобство split или метрики. Каждое расширение бюджета и фактическая причина остановки остаются в manifest. Если достигнут только минимальный объем, явно записать причину завершения этого этапа и не заявлять достигнутую цель 3 000.

Такой набор — ограниченная выборка EPA configurations, **не census** и не выборка продаж. Inclusion probabilities разных configurations не равны: round-robin дает сходное число raw downloads на активную пару year/make, меню и число комплектаций различаются, а scope filtering меняет доли. Нельзя обобщать средний расход или MAE на весь рынок без дополнительного обоснования. Полноту manufacturer/year coverage подтверждает отчет, а не одно отсутствие сетевых ошибок.

## Raw integrity и полнота

- [ ] Все retained IDs связаны с options menu и individual record; year/make/model совпадают.
- [ ] Raw files, request manifests и cache entries имеют проверяемые SHA-256, HTTP status, URL и UTC timestamp.
- [ ] Все failed, unfetched, conflicting-provenance records и недоступные меню перечислены; они не названы успешными downloads.
- [ ] Не перепутаны counts HTTP attempts, successful responses, menus, discovered IDs, downloaded vehicles и cleaned rows.
- [ ] Для bounded run явно указаны partial status, бюджет, stop reason и scope; не заявлена полнота catalogue traversal.
- [ ] Coverage показан для каждого 2015–2025 года и manufacturer, включая eligible count после filters; отсутствующие группы объяснены.

## Категории, топливо и hybrid metadata

- [ ] Просмотрен `category_inventory.csv` для **всех parseable raw records**, включая excluded; review итоговой таблицы недостаточен.
- [ ] Каждому наблюдавшемуся `VClass` присвоено include/exclude/needs-review с основанием. Cars, station wagons и четыре SUV класса проверены; pickups, vans и minivans не включены автоматически.
- [ ] `fuelType1` соответствует точной gasoline allowlist; значения смешанного/альтернативного топлива и непустой `fuelType2` не приняты через поиск слова gas.
- [ ] Проверены все наблюдавшиеся `atvType`, `evMotor`, `phevBlended`; missing key, null и неизвестный boolean не трактуются как подтвержденное отсутствие hybrid.
- [ ] Непустые hybrid/mild-hybrid metadata исключены; сигналы HEV/PHEV/MHEV/hybrid/eAssist в model/engine text проверены. Один `startStop=Y` не считается hybrid.
- [ ] Compatibility aliases `atvtype`/`basemodel` и конфликтующие варианты отражены в flags/exclusions.
- [ ] Отдельно просмотрены accepted records с необычным engine description/powertrain metadata и все неизвестные значения. При новых правилах очистки создана версия config до split; исторический frozen config не переписан.

## Target и missingness

- [ ] Target рассчитан только как `235.2145833333333 / comb08`; finite и positive. Нет построчного fallback к `comb08U`, target imputation или предварительного rounding.
- [ ] Missing/invalid targets исключены с reason codes. Высокий расход и большие двигатели не удалены ради метрик.
- [ ] Missing displacement/cylinders/transmission/drivetrain описаны с counts; invalid numeric specs и отсутствие обоих engine numeric fields обработаны по contract.
- [ ] Независимые reason totals отделены от sequential before/after counts; пересекающиеся причины не суммируются как отдельные автомобили.
- [ ] Принятый dataset не содержит rows с integrity failure или quarantine reason. Все новые категории/аномалии имеют documented decision.

## Семантические duplicates

- [ ] `vehicle_id` уникален; повторная загрузка ID не является дополнительной строкой.
- [ ] Просмотрены `duplicate_candidates.csv` и `duplicate_resolution.csv`. Кандидаты сформированы по идентификаторам технической конфигурации, без target в ключе.
- [ ] Одинаковые семь X не приняты за duplicate: разные engine/trim details могут быть законными конфигурациями.
- [ ] Автоматическая complete-source equivalence проверена по raw; удалены только известные ID/date metadata. Representative и mapping сохранены детерминированно.
- [ ] Для каждой группы с raw differences/target conflicts вынесено documented решение retained distinct / confirmed duplicate / quarantine. Расходы не усреднены, ID не выбраны по удобному target.
- [ ] Если остаются unresolved candidates, distinct count назван provisional. Их число, влияние на count и границы утверждения о >=1 000/>=3 000 показаны явно; не выдан автоматический проход cleaner за законченный semantic audit.

## Семейства и aliases до split

- [ ] Просмотрены inventory `manufacturer + baseModel`, fallback full model names и potential aliases без выбора по target/метрикам.
- [ ] Exact normalized fallback/baseModel matches проверены; остальные merges обоснованы raw identity/каталогом, а не сокращением до первого слова.
- [ ] Для review решений создан alias CSV с `manufacturer,model_name,canonical_base_model`; отдельно сохранены основания каждого merge.
- [ ] Связанные варианты AWD/2WD, transmission/engine suffixes, model spelling across years и отсутствие `baseModel` просмотрены. Одинаковое название у разных manufacturers не объединено.
- [ ] Необъясненные fallback/alias cases перечислены. Ограничение независимости unseen families явно отражено, если unresolved cases остались.
- [ ] Group mapping и его SHA-256 готовы до split; год не включен в group key. Smoke groups/folds не подставлены как основной benchmark.

## Готовность к freeze

- [ ] Достигнут заявленный volume после documented duplicate decisions; именно конфигурации, не число HTTP responses.
- [ ] Data card описывает фактический scope, sampling, exclusions, missingness, duplicates, target quantization, source dates и нерешенные ограничения.
- [ ] Raw-field dictionary покрывает все observed keys; неподтвержденные meanings помечены unreviewed, семь predictors не названы полным raw dictionary.
- [ ] Predictor allowlist проверена: economy/cost/emissions/scores/IDs/provenance/group/text не входят в Midterm X.
- [ ] Данные/config/code/dependency версии и hashes сохранены. После freeze расширение collection или cleaning оформляется отдельным benchmark.
- [ ] После split проверены exact retained-ID coverage, отсутствие train/test и CV group overlaps, непустые folds, hashes и coverage. Seed не перебирался ради target balance/test scores.

Итог audit: **pending**. Заполненный отчет должен отличать автоматические проверки, AI review, человеческий review и оставшиеся ограничения. Один unchecked пункт не надо скрывать формулировкой «все данные проверены».
