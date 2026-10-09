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

## Полный сбор

```powershell
python -m fuel_consumption.collect --config configs/project.json --snapshot full_2015_2025
```

Обход выполняется в два этапа: меню всех годов/марок/моделей, затем отдельные vehicle records. Полный сбор может занять часы. В любой момент можно остановить процесс и продолжить:

```powershell
python -m fuel_consumption.collect --config configs/project.json --snapshot full_2015_2025 --resume
```

Повторный запуск требует того же config и scope. Не изменять файл config в середине snapshot. Успешные bytes проверяются по SHA-256 и читаются из cache. Поврежденное доказательство не перезаписывается: ошибка требует проверки и нового snapshot. Подробности статуса находятся в `data/raw/full_2015_2025/snapshot.json`.

Для ограниченной проверки есть `--years`, `--makes`, `--max-models-per-make`, `--max-vehicles`, `--max-requests`, `--seed`. Такие runs честно отмечаются как partial. `--max-vehicles` — общее количество записей snapshot, `--max-requests` — новый бюджет запросов каждого invocation.

## Benchmark после полного сбора

```powershell
python -m fuel_consumption.clean --config configs/project.json --snapshot full_2015_2025
python -m fuel_consumption.split --config configs/project.json --dataset data/processed/vehicles.parquet
python -m fuel_consumption.train --config configs/project.json --dataset data/processed/vehicles.parquet --stage midterm --run midterm_v1
python -m fuel_consumption.evaluate --run midterm_v1
```

Cleaner сохраняет audit reports даже при недостаточном числе строк и завершает обычный запуск ошибкой, если минимум 1 000 не достигнут. До benchmark freeze проверить class/fuel inventories, missing powertrain metadata, semantic duplicates и aliases семейств. Автоматический полный обход не заменяет этот предметный аудит.

`split` сохраняет IDs и folds; следующий запуск на той же версии использует сохраненный manifest. Попытка переиспользовать его с измененными data/config отвергается. `train` выбирает модель по train CV до test prediction и сохраняет модели, CV selection, metrics и per-row predictions. Existing run не перезаписывается: использовать новый run ID.

## Технический пример

В `examples/smoke/` находится ограниченный реальный сбор, отдельная обработанная таблица, split и результаты. Он не выполняет минимальный объем Midterm. Чтобы воспроизвести очистку без сети:

```powershell
python -m fuel_consumption.clean --config configs/project.json --snapshot examples/smoke/raw --output-root work/reproduced_smoke --allow-small
python -m fuel_consumption.split --config configs/project.json --dataset work/reproduced_smoke/data/processed/vehicles.parquet --output work/reproduced_smoke/data/splits --allow-small
python -m fuel_consumption.train --config configs/project.json --dataset work/reproduced_smoke/data/processed/vehicles.parquet --split-dir work/reproduced_smoke/data/splits --output-root work/reproduced_smoke --run smoke_reproduced --allow-small
```

Для `--snapshot` путь к example удобнее передавать абсолютным, если CLI работает из другой папки. Notebook `notebooks/01_midterm.ipynb` запускается offline и заново обучает модели во временной папке; default dataset и split указывают на example. Для полноценного benchmark пересоздать его:

```powershell
python scripts/build_notebook.py --dataset data/processed/vehicles.parquet --split-dir data/splits
```

Перед запуском notebook в новой среде установить/выбрать kernel этого окружения. Никакой сетевой collection не выполняется при Run All.
