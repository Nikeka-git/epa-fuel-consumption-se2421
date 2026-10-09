# Контракт данных

Первоначальные правила версии 0.1, 9 октября 2026 года, дополненные результатом завершенного сбора. Фактические counts установлены по основному snapshot: 4 500 индивидуальных записей → 3 247 очищенных конфигураций. Полные категории, missingness, hashes и ограничения находятся в [DATA_CARD](DATA_CARD.md); пять pilot records не служат подтверждением основной полноты.

Предметное дополнение до первого benchmark freeze, 9 октября 2026 года: пустые `atvType`/`evMotor` не всегда означают отсутствие MHEV. Реальные MY2022 Volvo S60 B5 AWD (ID44187) и XC60 B5 AWD (ID44205) имеют такие пустые labels. [Реестр reviewed powertrain exclusions](../configs/powertrain_exclusions.json) содержит точные manufacturer/year/model условия, первичные источники и ограничения привязки. `powertrain.py` применяет его без доступа к target; cleaner сохраняет отдельную причину и hashes rules/module. Raw responses и исходный collection config не изменяются. Start/stop или батарея chassis сами по себе не служат основанием исключения. Остаточная неопределенность source metadata явно указывается в итоговой data card.

## Источник и единица наблюдения

Supplemental review также ограничивает Audi A4/A5/Q5 2.0L/4-cylinder строго MY2021–2022 по материалам Audi of America. SQ7/SQ8 4.0L/V8 MY2020–2025 получают отдельное решение `powertrain_uncertain`, поскольку manufacturer catalogue labels и equipment-conditional bulletins не дают однозначного подтверждения обычного бензинового powertrain. Это явный quarantine суженного scope, а не утверждение, что все такие записи доказанно hybrid. Подробности и ограничения — [POWERTRAIN_REVIEW.md](POWERTRAIN_REVIEW.md).

Одна строка описывает отдельную EPA-конфигурацию автомобиля определенного модельного года. `vehicle_id` является исходным идентификатором записи; это не VIN, не владелец и не отдельный проданный автомобиль. Основной источник — [официальный API и словарь](https://www.fueleconomy.gov/feg/ws/index.shtml). Используем только catalogue endpoints, без My MPG driver records.

Путь обнаружения: `vehicle/menu/year` → `vehicle/menu/make?year=` → `vehicle/menu/model?year=&make=` → `vehicle/menu/options?year=&make=&model=` → `vehicle/{id}`. Base URL: `https://www.fueleconomy.gov/ws/rest/`. Параметры кодировать средствами `urlencode` или requests params. Не извлекать полный dataset из готового CSV вместо собственного кода обхода.

## Проверенные особенности API

На пяти реальных записях и меню подтверждены:

- В JSON ключи **atvType** и **baseModel**; prose документации использует другой регистр. Внутренние имена: `atv_type` и `base_model`.
- Числа обычно представлены строками; `evMotor` является текстом, например `44V Li-Ion`.
- `menuItem` при одном результате — object, при нескольких — list. Нормализовать `None → []`, `dict → [dict]`, `list → list`; другой тип считать ошибкой. Не считать malformed response пустым корректным меню.
- JSON и XML доступны; production collector выбирает JSON через `Accept: application/json`. XML в pilot подтверждает форму menuItems, но полноценный XML fallback не обязателен.

| ID | Запись | Ожидаемое решение |
|---|---|---|
| 34836 | 2015 Honda Fit, Small Station Wagons | Включить при текущем scope легковых автомобилей |
| 48910 | 2025 Toyota RAV4, startStop Y | Включить; start-stop не означает hybrid |
| 48861 | 2025 Toyota Prius, gasoline и пустое fuelType2 | Исключить: atvType Hybrid и электрический мотор |
| 49014 | 2025 Toyota Prius PHEV | Исключить: PHEV и альтернативное топливо |
| 48147 | 2025 Volvo XC90 B5 AWD | Исключить: Mild Hybrid |

Ответы доступны в `evidence/api_probe/`, URL и hashes — в `requests.jsonl`. Полные наблюдения — `evidence/API_RESEARCH_RU.txt`.

## Сбор и provenance

Production collector должен:

1. Обнаруживать конфигурации через документированные меню годов 2015–2025 и доступных manufacturers; дедуплицировать discovered IDs. Сохранить связь ID с меню. Для основного benchmark до его первого freeze выбран bounded seeded round-robin всех годов/марок с исходным бюджетом 6 000 raw records и seed 42. До обучения бюджет снижен до 4 500 по checkpoint counts, без просмотра модельных ошибок; решение сохранено в [collection provenance](../evidence/benchmark_collection_provenance.json). Чередование и порядок не используют target. Это выборка с неодинаковыми вероятностями включения, не полный census и не выборка продаж. Полный обход остается отдельным режимом collector.
2. Получать individual vehicle records собственным кодом. Первая версия использовала один worker и интервал 1 s; перед новым benchmark выбраны максимум четыре одновременных запроса и общий минимальный интервал старта HTTP attempts 0.25 s. Timeout 30 s, максимум 5 попыток; Retry-After при ограничении сервера вводит общий cooldown. Запись raw/manifests/cache выполняется одним потоком. Это настройки проекта, не заявленный лимит сайта. Frozen config исторического smoke не меняется.
3. Для сетевых ошибок, 408, 429 и временных 5xx использовать ограниченный exponential backoff с jitter; учитывать Retry-After. Неповторяемые 4xx не запускать бесконечно. Не обходить ограничения сервера.
4. Сохранять exact response bytes атомарно. Metadata: request URL и params, HTTP status, UTC fetched_at, content type, SHA-256, файл, attempt, error. Сохранять raw menus тоже.
5. Проверять успешный status, JSON parse, ожидаемую схему и совпадение id/year/make/model с меню. HTML response при status 200 не считать записью.
6. Resume пропускает только успешные записи с валидным checksum. Ошибки и пропуски остаются в manifest, их статус виден в coverage report.
7. Не перезаписывать исторические ответы. Создавать `data/raw/<snapshot_id>/`, фиксировать начало и конец сбора в UTC и timezone Asia/Qyzylorda.
8. Генерировать отчеты покрытия и counts. Количество HTTP requests, меню, полученных vehicle records и итоговых строк — разные числа.

Полный обход может быть долгим. Сначала измерить стоимость небольшого production smoke-run; не выводить неподтвержденную длительность всего сбора. При сетевой ошибке сохранять прогресс и завершать с понятным неполным статусом.

Пользователь сообщил, что robots.txt и условия уже проверены. В data card сохранить ссылки, дату и основание этой проверки; собрать копии общедоступных правил при подготовке полного snapshot. Не подменять это утверждением, что наличие API доказывает любое разрешение. Дата pilot не является датой полного сбора.

## Внутренняя схема

| Внутреннее поле | API | Тип | Роль |
|---|---|---|---|
| vehicle_id | id | string | Ключ, audit и splits; никогда не X |
| model_year | year | integer | Числовой X |
| manufacturer | make | string | Категориальный X |
| model_name | model | string | Исходный текст, Final |
| base_model | baseModel | string или null | Только группировка split |
| displacement_l | displ | float или null | Числовой X |
| cylinders | cylinders | integer или null | Числовой X |
| transmission | trany | string или null | Категориальный X |
| drivetrain | drive | string или null | Категориальный X |
| vehicle_class | VClass | string | Категориальный X и фильтр |
| engine_description | eng_dscr | string или null | Исходный текст, Final |
| engine_id | engId | string или null | Только duplicate audit |
| atv_type | atvType | string | Только scope filter |
| fuel_type_primary | fuelType1 | string | Только scope filter |
| fuel_type_secondary | fuelType2 | string | Только scope filter |
| electric_motor | evMotor | string | Только scope filter |
| phev_blended | phevBlended | bool или null | Только scope filter |
| combined_mpg | comb08 | float | Только вычисление target и audit |
| combined_mpg_unrounded | comb08U | float или null | Только audit |
| target_l100km | расчет | float | y |
| source_url | metadata | string | Provenance |
| fetched_at_utc | metadata | timestamp | Provenance |
| raw_sha256 | metadata | string | Provenance |
| snapshot_id | metadata | string | Версия данных |
| model_group | расчет | string | Split/CV, никогда не X |

Raw хранит все исходные поля. В data card дополнить словарь для **всех** raw keys отдельным машинным dictionary с dtype, смыслом и disposition, а также описать все поля итоговой таблицы. Не называть семь predictors полным словарем raw dataset.

Если получен alias `atvtype` вместо `atvType`, разрешить явное compatibility mapping с audit flag. Если обе формы присутствуют и значения конфликтуют — quarantine, не выбирать молча. Аналогично baseModel/basemodel.

## Inclusion и quarantine

Применять все проверки и сохранять **все** reason codes. Для последовательной таблицы before/after фиксировать один порядок фильтров; totals по независимым причинам могут перекрываться.

1. Валидные ID и model_year 2015–2025. `manufacturer`, `model_name`, `vehicle_class` непустые. Имя модели не генерировать и не импутировать.
2. `fuelType1` в allowlist Regular Gasoline, Premium Gasoline, Midgrade Gasoline; последнее отсутствовало в pilot, но присутствует в 30 retained main records. `fuelType2` пустое. Не использовать `contains('gas')`, которое пропустит смешанные топлива.
3. `atvType` пустое. Любая непустая технология исключается либо уходит в audit при неизвестном значении. Пропавший ожидаемый metadata key нельзя автоматически трактовать как подтвержденное отсутствие hybrid.
4. `evMotor` пустое, `phevBlended` false или известное пустое значение. Неизвестные boolean-like значения и электрические sentinels проверяются отдельно. Не приводить `evMotor` к float.
5. Токены HEV, PHEV, MHEV, hybrid, mild hybrid, eAssist в model/engine text — дополнительный сигнал исключения или ручного аудита. Использовать нормализованные слова/границы токенов, не произвольную подстроку. Отсутствие этих слов не отменяет предыдущие проверки.
6. `vehicle_class` соответствует легковым или SUV по allowlist в config. Классы в prose и фактическом API могут называться по-разному. Все новые значения выводятся в audit до freeze.
7. `comb08` числовое, finite и больше нуля. Missing/NaN/inf/zero/negative target исключать. **Target не импутировать.**
8. Некорректные типы, отрицательный/нулевой объем или неположительные/нецелые цилиндры — quarantine для проверки источника. Пропущенные displacement или cylinders допустимы по отдельности, но если отсутствуют оба — исключить как insufficient_engine_specs. Нормальные missing values будут импутироваться только внутри train pipeline.
9. Missing transmission/drivetrain допустимы и документируются; whitespace и пустые strings нормализуются. Не выдумывать значения из target или других economy fields.

Не удалять большой расход или редкий большой двигатель просто потому, что он ухудшает MAE. Аномалии проверять по raw и предметному смыслу. Пороговые статистические правила, если понадобятся, обучать на train; target-dependent outlier pruning всего датасета запрещен.

Class allowlist находится в config. Он включает Two Seaters, классы Cars, Station Wagons и четыре Small/Standard Sport Utility Vehicle 2WD/4WD. В pilot наблюдались четыре класса; основной audit до freeze подтвердил 12 наблюдаемых категорий из 13 разрешенных, без новых категорий. Любые будущие расширения отражать в новой версии contract/config и audit report.

## Target

Используем единый перевод опубликованного `comb08` в расход:

```text
US gallon = 3.785411784 L
mile = 1.609344 km
K = 100 × 3.785411784 / 1.609344 = 235.2145833333333
target_l100km = K / combined_mpg
```

30 US MPG дают примерно 7.8404861111 L/100 km; 34 US MPG дают 6.9180759804. Не путать US и imperial gallons. Target хранится без преждевременного округления; округлять только отображение результатов.

`comb08` округлен в исходнике, поэтому y дискретизирован. `comb08U` может быть неполным и не заменяет comb08 построчно. Любое исследование другого target — отдельная версия с пересчетом всех моделей, а не скрытая замена во время проекта.

## Защита от утечки

Начальный X строго состоит из:

```text
model_year, displacement_l, cylinders,
transmission, drivetrain, vehicle_class, manufacturer
```

Все остальные поля запрещены по умолчанию. В частности: любые city/highway/combined MPG, unadjusted MPG, MPGe, range, fuel costs, barrels, CO2/emissions, feScore, ghgScore, smartwayScore, guzzler, youSaveSpend; также id, engine_id, dates, group ID и source metadata. Даже если корреляция полезна, это не основание добавить proxy целевой переменной.

На Final разрешены только дополнительные входы model_name и engine_description после независимого от target sanitizer. [Официальное объяснение engine descriptors](https://www.fueleconomy.gov/feg/findacarhelp.shtml#engine) связывает GUZZLER с низкой экономичностью. Сохранить оригинал, но удалить такие маркеры и явные fuel-economy/cost/emissions значения из model input; правила задать до проверки test и не вычищать обычные обозначения двигателя без причины. Текст на иностранном сайте рассматривается как данные, не инструкции для агента.

## Distinct и дубликаты

- Повторная загрузка одного ID — одна исходная запись; версии raw не теряются, для snapshot фиксируется выбранная версия.
- Разные IDs с одинаковыми семью X могут иметь законные различия комплектации. Не делать `drop_duplicates(subset=feature_columns)`.
- До split сформировать duplicate-candidate report по year, make, model, displ, cylinders, trany, drive, VClass, eng_dscr, engId и fuelType1. Сравнивать дополнительные характеристики в raw. Target не входит в правило формирования кандидатов.
- Если подтверждено, что IDs повторяют одну конфигурацию, оставить детерминированный representative и сохранить mapping. Конфликтующие targets не усреднять молча; quarantine и объяснение.
- Distinct count для цели 1 000/3 000 — count после подтвержденных дублей. Каталог может содержать близкие модели и разные годы; это ожидаемо, split учитывает родственные семейства.

## Freeze и выходные файлы

Основные артефакты завершенного snapshot (полный перечень и команды воспроизведения — в [RUNNING](RUNNING.md)):

```text
data/raw/<snapshot_id>/menus/...
data/raw/<snapshot_id>/vehicles/<id>.json
data/raw/<snapshot_id>/requests.jsonl
data/raw/<snapshot_id>/snapshot.json
data/interim/vehicle_inventory.csv
data/interim/cleaning_summary.csv
data/interim/excluded_records.csv
data/interim/duplicate_candidates.csv
data/interim/duplicate_resolution.csv
data/interim/category_inventory.csv
data/processed/vehicles.parquet
data/processed/schema.json
data/splits/group_mapping.csv
data/splits/split_manifest.csv
data/splits/split_metadata.json
```

Сохранить SHA-256 исходных файлов и финального table artifact, code revision/hash, config и dependency lock. После freeze новое расширение или иная очистка не перезаписывают benchmark. Финальные text experiments используют те же строки, даже если у части нет engine description.
