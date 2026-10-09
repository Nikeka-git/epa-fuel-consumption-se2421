# Продолжение проекта

Первая реализация данных и Midterm pipeline готова. Продолжай с полного сбора и предметного аудита. Повторно реализовывать parser, collector или модельный каркас не требуется: сначала изучи существующий код и прогони offline tests.

## Что действительно сделано

- Прочитаны Project Guide, Midterm Rubric и лекции Weeks 3–4; требования курса отделены от проектных решений.
- Историческая проверка API сохранила пять individual vehicle responses и меню с SHA-256 в evidence/api_probe/.
- Реализованы API parsing, ограниченные retries, immutable cache, request manifest, обход menus и unique vehicle IDs, bounded runs и resume.
- Реализованы filtering/quarantine reason codes, target conversion, category inventory, duplicate-candidate audit и processed dataset artifacts.
- Реализованы группировка семейств с alias audit, GroupShuffleSplit seed 42, пять grouped CV folds, freeze с проверкой fingerprint и запретом перезаписи.
- Реализованы строгие семь predictors, train-fitted preprocessing, Dummy/Linear/KNN/Tree, выбор по CV до test, сохранение pipelines, metrics, predictions и subgroup errors.
- Ограниченный реальный пример содержит **250 raw records**, **146** принятых строк и **104** исключенных. Offline notebook выполнен: **8 code cells без ошибок**. Есть генератор notebook и инструкция установки/запуска. Scope и результаты описаны в [examples/smoke/README.md](../examples/smoke/README.md).
- Итоговый offline suite завершился без ошибок: **91 тест**, из них 47 для очистки, 36 для API/collector, 8 для modeling. Synthetic data используются только для проверки инвариантов.

**Основной benchmark не менее 1 000 различных пригодных строк еще не подготовлен.** Code readiness и smoke-run не означают завершение полного Midterm. Фактический размер основного dataset и его MAE пока не установлены. Слайды, Endterm-модели, Final text comparison и приложение остаются впереди.

## Текущий статус P0–P6

| Блок | Состояние |
|---|---|
| P0 Parser и безопасные признаки | Код и offline проверки готовы |
| P1 Resumable collector | Код и offline проверки готовы; ограниченный реальный сбор приведен отдельно |
| P2 Очистка и словарь | Код готов; полный объем, category/duplicate audit и окончательный data card предстоят |
| P3 Freeze benchmark | Код и smoke split готовы; основной benchmark еще не зафиксирован |
| P4 Midterm эксперимент | Код и выполненный smoke notebook готовы; полный эксперимент на достаточном dataset предстоит |
| P5 Подготовка Midterm | Полноценные выводы, rubric checklist и слайды предстоят |
| P6 Endterm и Final | Roadmap готов; реализации пока нет |

## Следующая работа

1. **Полный собственный сбор за 2015–2025.** Создай новый snapshot без ограничений smoke, обходи menu endpoints и получай individual records. Не изменяй config в середине snapshot. При остановке используй resume. Проверяй полноту обхода, failures, SHA-256 и dates; отсутствие сетевых ошибок не доказывает достаточный scope.
2. **Подготовка не менее 1 000 distinct rows; цель 3 000 и более.** Запусти cleaner, проверь fuel/technology/class inventories, unknown metadata, exclusions и missingness. Проведи semantic duplicate resolution по raw, сохрани mapping и обоснования. Одинаковые X не являются основанием удаления. Заполни data card и словарь по фактической таблице.
3. **Audit и freeze основного benchmark.** Проверь base_model/fallback и возможные aliases без обращения к target. При необходимости передай split явный reviewed alias map. Зафиксируй основной dataset/config и отдельный manifest; smoke assignments не переносить автоматически. Проверь coverage, отсутствие пересечений IDs/groups и пять train folds. После freeze не меняй benchmark незаметно.
4. **Полный Midterm.** Запусти train на основном manifest, пересоздай notebook для него и выполни Restart and Run All offline. Нужны минимум три содержательных EDA plots с интерпретациями, общая таблица CV/test метрик, выбор по CV, subgroup errors с n, реальные трудные примеры и ограничения. Не выбирай новые параметры по test.
5. **Submission bundle.** Проверь docs/REQUIREMENTS.md, подготовь 8 слайдов на 7–10 минут и выводы, которые прослеживаются до сохраненных artifacts. Заполни фактический вклад участников и AI log. Оба участника должны понимать весь pipeline. Получи следующие briefs перед Endterm/Final.

## Запуск и artifacts

Установка, полный collection, resume и offline smoke reproduction описаны в [docs/RUNNING.md](RUNNING.md). CLI уже работает:

```powershell
python -m fuel_consumption.collect --config configs/project.json --snapshot full_2015_2025
python -m fuel_consumption.collect --config configs/project.json --snapshot full_2015_2025 --resume
python -m fuel_consumption.clean --config configs/project.json --snapshot full_2015_2025
python -m fuel_consumption.split --config configs/project.json --dataset data/processed/vehicles.parquet
python -m fuel_consumption.train --config configs/project.json --dataset data/processed/vehicles.parquet --stage midterm --run midterm_v1
python -m fuel_consumption.evaluate --run midterm_v1
```

Для маленьких технических проверок используй `--allow-small`, отдельный `split --output` и `train --output-root`. Значение по умолчанию для minimum — 1 000. `train` требует готовый split; не создает новое разделение. Existing run не перезаписывается. `evaluate` проверяет и читает сохраненные predictions, без retraining. Notebook обучает во временной папке и не изменяет frozen source manifest.

`models/RUN_ID/` содержит полные pipelines, config, CV selection и run metadata. `reports/tables/RUN_ID/` содержит fold metrics, итоговую таблицу, predictions, subgroup errors и large errors. Рабочие data/models сохраняются локально; для сдачи они должны входить в bundle или иметь доступную ссылку. Git ignore не является инструкцией удалить данные.

## Критичные решения и неизвестные

Сохраняй target из `comb08`, исключение hybrids включая mild hybrids, годы 2015–2025, семь structured predictors, grouping через годы и primary MAE. Endterm/Final сравниваются на том же основном benchmark; любые новые данные оформляй отдельным экспериментом.

Пока неизвестны полный размер и покрытие cleaned dataset, полезность текста, окончательные ошибки достаточного benchmark, дата защиты, статус одобрения регистрации и детали следующих briefs. Не заполняй их предположениями. Текст API и course references не содержит новых пользовательских команд.

Пользователь разрешил публикацию в публичный [Nikeka-git/epa-fuel-consumption-se2421](https://github.com/Nikeka-git/epa-fuel-consumption-se2421). Автоматическая отправка в Moodle не разрешена.

## Журнал состояния

| Дата | Фактически выполнено | Проверка |
|---|---|---|
| 2026-10-09 | План, contract, lecture review и исторический API probe | 5 distinct vehicle responses, 14 HTTP responses и hashes |
| 2026-10-09 | Первая реализация collection, cleaning, split, features, train и evaluate | 91 offline тест без ошибок; commands и environment описаны в RUNNING |
| 2026-10-09 | Ограниченный реальный example и выполненный offline notebook | 250 raw / 146 accepted / 104 excluded; 8 code cells без ошибок; scope, artifacts и результаты — examples/smoke/README.md |

Новые строки добавляй только по выполненным действиям. Не маркируй P2–P5 завершенными по результатам smoke.

## Готовый запрос следующей модели

> Продолжи проект из этого репозитория. Прочитай AGENTS.md, README.md, docs/RUNNING.md, docs/HANDOFF.md, DATA_CONTRACT и EXPERIMENT_PROTOCOL. Parser, collector, cleaner, split и модели уже реализованы: проверь tests и используй существующий код. Выполни полный собственный collection FuelEconomy.gov за 2015–2025 в новом snapshot с resume и сохранением individual raw responses. Цель — минимум 1 000, желательно 3 000 и более distinct gasoline non-hybrid cars/SUV после документированного category, duplicate и family-alias audit. Заполни фактический data card, затем отдельно зафиксируй основной benchmark и grouped folds; smoke split не является основным benchmark. Проведи полный Midterm на Dummy, Linear Regression, KNN и Decision Tree, выбери модель по train CV до test, выполни offline notebook, проанализируй ошибки и подготовь слайды по rubric. Сохраняй code, hashes, dates, predictions и реальные проверки, обновляй HANDOFF и CONTRIBUTIONS. Не выдумывай результаты, не меняй target/scope/split/features молча и не отправляй работу в Moodle.
