# Запуск проекта

Все команды выполняются из корня репозитория. Проверенное окружение первой реализации: Python 3.12; точные версии библиотек находятся в requirements.lock.txt. Сетевой сбор использует стандартную библиотеку; очистка и моделирование требуют установленного пакета.

## Установка в PowerShell

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock.txt
.venv/Scripts/python.exe -m pip install -e . --no-deps
.venv/Scripts/python.exe -m pytest -q
```

На Linux/macOS заменить `.venv/Scripts/python.exe` на `.venv/bin/python`. Для обычной разработки без воспроизведения точных версий доступно `python -m pip install -e '.[dev]'`.

## Основной benchmark: ограниченная выборка всех годов и марок

До первого freeze выбран бюджет 6 000 исходных записей, все 11 лет и все доступные manufacturers, seed 42. Это выборка каталога, не полный census и не набор, взвешенный по продажам. Порядок не использует величины target. Настройки основного сбора: четыре HTTP workers, общий интервал старта attempts не менее 0.25 s, общий cooldown по Retry-After, единственный writer raw/manifests/cache.

```powershell
python -m fuel_consumption.collect --config configs/project.json --snapshot benchmark_2015_2025_20261009 --seed 42 --max-vehicles 6000
python -m fuel_consumption.collect --config configs/project.json --snapshot benchmark_2015_2025_20261009 --seed 42 --max-vehicles 6000 --resume
```

Вторая команда используется после остановки первой. Не менять config в середине snapshot. История invocation, dates, scope и budgets сохранены. Resume проверяет уже опубликованные bytes. Повторный live-сбор не обещает побайтово одинаковую выборку: источник изменяется, а прежняя версия resume начинала обход с первой пары. Для воспроизводимости моделирования использовать опубликованные immutable raw, cleaned dataset и сохраненный manifest IDs/groups/folds.

## Неограниченный census как отдельный режим

```powershell
python -m fuel_consumption.collect --config configs/project.json --snapshot full_2015_2025
```

Обход выполняется в два этапа: меню всех годов/марок/моделей, затем отдельные vehicle records. Полный сбор может занять часы. В любой момент можно остановить процесс и продолжить:

```powershell
python -m fuel_consumption.collect --config configs/project.json --snapshot full_2015_2025 --resume
```

Повторный запуск требует того же config и scope. Не изменять файл config в середине snapshot. Успешные bytes проверяются по SHA-256 и читаются из cache. Поврежденное доказательство не перезаписывается: ошибка требует проверки и нового snapshot. Подробности статуса находятся в `data/raw/full_2015_2025/snapshot.json`.

Для ограниченной проверки есть `--years`, `--makes`, `--max-models-per-make`, `--max-vehicles`, `--max-requests`, `--seed`. Такие runs честно отмечаются как partial. `--max-vehicles` — общее количество записей snapshot, `--max-requests` — новый бюджет запросов каждого invocation.

## Обработка и аудит до freeze

```powershell
python -m fuel_consumption.clean --config configs/project.json --snapshot benchmark_2015_2025_20261009
python scripts/build_field_dictionary.py --interim data/interim
python scripts/audit_benchmark.py --snapshot benchmark_2015_2025_20261009 --dataset data/processed/vehicles.parquet --dictionary data/interim/raw_field_dictionary.json --output reports/audit/benchmark_2015_2025_20261009
```

Проверить audit до следующего блока. Integrity issues, неизвестные scope metadata, raw differences кандидатов и family aliases требуют явного решения. Утилита аудита ничего не удаляет и не угадывает aliases. Недокументированные дополнительные raw fields явно помечены в dictionary и исключены из predictors; семь inputs имеют проверенные определения.

```powershell
python -m fuel_consumption.split --config configs/project.json --dataset data/processed/vehicles.parquet
python -m fuel_consumption.train --config configs/project.json --dataset data/processed/vehicles.parquet --stage midterm --run midterm_v1
python -m fuel_consumption.evaluate --run midterm_v1
```

Cleaner сохраняет audit reports даже при недостаточном числе строк и завершает обычный запуск ошибкой, если минимум 1 000 не достигнут. До benchmark freeze проверить class/fuel inventories, missing powertrain metadata, semantic duplicates и aliases семейств. Автоматический полный обход не заменяет этот предметный аудит.

`split` сохраняет IDs и folds; следующий запуск на той же версии использует сохраненный manifest. Попытка переиспользовать его с измененными data/config отвергается. `train` выбирает модель по train CV до test prediction и сохраняет модели, CV selection, metrics и per-row predictions. Existing run не перезаписывается: использовать новый run ID.

## Notebook, отчет и слайды

После завершенного run:

```powershell
python scripts/build_notebook.py --run-dir models/midterm_v1
python scripts/build_midterm_report.py --run-dir models/midterm_v1 --output-dir reports/midterm/midterm_v1
python scripts/prepare_slide_summary.py --dataset data/processed/vehicles.parquet --split-dir data/splits --tables reports/tables/midterm_v1 --snapshot data/raw/benchmark_2015_2025_20261009 --config configs/project.json --output reports/slides/slide_summary.json
```

Notebook в режиме `--run-dir` читает и проверяет сохраненные результаты без retraining. Без этого аргумента notebook заново обучает четыре модели во временной папке. Report generator сохраняет Markdown, четыре scientific figures и таблицы; не перезаписывает существующий report directory. После генерации выполнить notebook сверху вниз в kernel установленного проекта, сохранить все outputs.

`scripts/build_midterm_slides.mjs` создает editable PPTX из slide summary и использует bundled `@oai/artifact-tool` и finalizer Codex; это дополнительная среда для сборки слайдов, не зависимость collection/ML pipeline. Готовые слайды читаются в обычном PowerPoint. Восемь слайдов и speaker notes рассчитаны на 9 минут 20 секунд; реальные репетиция и распределение речи участниками еще нужны.

## Технический пример

В `examples/smoke/` находится ограниченный реальный сбор, отдельная обработанная таблица, split и результаты. Он не выполняет минимальный объем Midterm. Чтобы воспроизвести очистку без сети:

```powershell
python -m fuel_consumption.clean --config examples/smoke/models/first_20261009/config.json --snapshot examples/smoke/raw --output-root work/reproduced_smoke --allow-small
python -m fuel_consumption.split --config examples/smoke/models/first_20261009/config.json --dataset work/reproduced_smoke/data/processed/vehicles.parquet --output work/reproduced_smoke/data/splits --allow-small
python -m fuel_consumption.train --config examples/smoke/models/first_20261009/config.json --dataset work/reproduced_smoke/data/processed/vehicles.parquet --split-dir work/reproduced_smoke/data/splits --output-root work/reproduced_smoke --run smoke_reproduced --allow-small
```

Для `--snapshot` путь к example удобнее передавать абсолютным, если CLI работает из другой папки. Исторический example сохраняет свой config; новый root config ему не подставлять. Генератор по умолчанию использует основной dataset/split. Для самостоятельного refit notebook основного benchmark:

```powershell
python scripts/build_notebook.py --dataset data/processed/vehicles.parquet --split-dir data/splits
```

Перед запуском notebook в новой среде установить/выбрать kernel этого окружения. Никакой сетевой collection не выполняется при Run All.
