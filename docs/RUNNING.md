# Запуск проекта

Все команды выполняются из корня репозитория в активированном окружении проекта. Проверенное окружение: Python 3.12; точные версии библиотек находятся в requirements.lock.txt. Сетевой сбор использует стандартную библиотеку; очистка и моделирование требуют установленного пакета. Это демонстрационный проект проверки ИИ; команды воспроизводят pipeline, а не подтверждают фактическую университетскую сдачу или защиту.

## Установка в PowerShell

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock.txt
.venv/Scripts/python.exe -m pip install -e . --no-deps
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/Activate.ps1
```

На Linux/macOS заменить `.venv/Scripts/python.exe` на `.venv/bin/python` и активировать окружение командой `source .venv/bin/activate`. Для обычной разработки без воспроизведения точных версий доступно `python -m pip install -e '.[dev,app]'`: extra `app` устанавливает Streamlit. Если окружение не активировано, заменить `python` в следующих командах на путь к его интерпретатору.

## Основной benchmark: ограниченная выборка всех годов и марок

Основной benchmark уже завершен и включен в репозиторий: 4 500 исходных записей, 3 247 очищенных строк и все три model runs. Начальный бюджет 6 000 сокращен до 4 500 по числу пригодных записей до обучения; решение сохранено в evidence/benchmark_collection_provenance.json. Это выборка каталога, не полный census и не набор, взвешенный по продажам. Порядок не использует величины target. Настройки сбора: четыре HTTP workers, общий интервал старта attempts не менее 0.25 s, общий cooldown по Retry-After, единственный writer raw/manifests/cache.

Для нового live-сбора использовать новое имя snapshot. Опубликованный `benchmark_2015_2025_20261009` после freeze не возобновлять и не изменять:

```powershell
python -m fuel_consumption.collect --config configs/project.json --snapshot new_2015_2025 --seed 42 --max-vehicles 4500
python -m fuel_consumption.collect --config configs/project.json --snapshot new_2015_2025 --seed 42 --max-vehicles 4500 --resume
```

Вторая команда используется после остановки первой. Не менять config в середине snapshot. История invocation, dates, scope и budgets сохранены. Resume проверяет уже опубликованные bytes. Повторный live-сбор не обещает побайтово одинаковую выборку: источник изменяется, resume начинает logical обход с первой пары, а история остановок может менять prefix. Для воспроизводимости моделирования использовать опубликованные immutable raw, cleaned dataset и сохраненный manifest IDs/groups/folds.

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

Следующие команды показывают выполненный первоначальный workflow. Для существующего benchmark повторная очистка и новый split не нужны. Чтобы отдельно воспроизвести очистку опубликованных raw без сети, используйте другой output root:

```powershell
python -m fuel_consumption.clean --config configs/project.json --snapshot benchmark_2015_2025_20261009 --output-root work/replayed_data
```

```powershell
python -m fuel_consumption.clean --config configs/project.json --snapshot benchmark_2015_2025_20261009
python scripts/build_field_dictionary.py --interim data/interim
python scripts/audit_benchmark.py --snapshot benchmark_2015_2025_20261009 --dataset data/processed/vehicles.parquet --dictionary data/interim/raw_field_dictionary.json --output reports/audit/benchmark_2015_2025_20261009
```

Проверить audit до следующего блока. Integrity issues, неизвестные scope metadata, raw differences кандидатов и family aliases требуют явного решения. Утилита аудита ничего не удаляет и не угадывает aliases. Недокументированные дополнительные raw fields явно помечены в dictionary и исключены из predictors; семь inputs имеют проверенные определения.

```powershell
python -m fuel_consumption.split --config configs/project.json --dataset data/processed/vehicles.parquet --alias-map reports/audit/benchmark_2015_2025_20261009/family_review/final/reviewed_family_aliases.csv
python -m fuel_consumption.train --config configs/project.json --dataset data/processed/vehicles.parquet --stage midterm --run midterm_v1
python -m fuel_consumption.evaluate --run midterm_v1
```

Cleaner сохраняет audit reports даже при недостаточном числе строк и завершает обычный запуск ошибкой, если минимум 1 000 не достигнут. До benchmark freeze проверить class/fuel inventories, missing powertrain metadata, semantic duplicates и aliases семейств. Автоматический полный обход не заменяет этот предметный аудит.

`split` сохраняет IDs и folds; следующий запуск на той же версии использует сохраненный manifest. Попытка переиспользовать его с измененными data/config отвергается. `train` выбирает модель по train CV до test prediction и сохраняет модели, CV selection, metrics и per-row predictions. Existing run не перезаписывается: использовать новый run ID.

После freeze **не менять даже поле `status` в первоначальном config**: сверяется SHA-256 всех исходных bytes, а не только список признаков. Stage configs `configs/endterm.json` и `configs/final.json` самостоятельны и не заменяют первоначальный config. Если root config уже изменен для нового эксперимента, указать точный первоначальный файл, чей checksum записан в `split_metadata.json`. Сохраненная копия config пригодна только при совпадении этого checksum.

## Повторное обучение в отдельных runs

Основные runs `midterm_v1`, `endterm_v1`, `final_v1` уже completed и не перезаписываются. Для повторного обучения на сохраненных данных/folds:

```powershell
python -m fuel_consumption.train --run midterm_replay
python -m fuel_consumption.endterm --run endterm_replay --midterm-run models/midterm_replay
python -m fuel_consumption.final --run final_replay
python scripts/build_project_report.py --runs midterm_replay endterm_replay final_replay --output reports/project_replay
```

Эти команды создают новые outputs. Уже раскрытый test остается сравнительным benchmark; повторение не превращает его в новую независимую проверку. Для чтения готового notebook и работы приложения повторное обучение не требуется.

## Endterm: ensembles, MLP и сегменты

После Midterm использовать тот же dataset и split:

```powershell
python -m fuel_consumption.endterm --dataset data/processed/vehicles.parquet --split-dir data/splits --config configs/project.json --stage-config configs/endterm.json --run endterm_v1 --midterm-run models/midterm_v1
```

Отдельный stage config задает четыре кандидата Random Forest, четыре Extra Trees и два MLP. Пять outer folds взяты из исходного training manifest; внутри каждого идет train-only GroupKFold search. Каждая pipeline обучает собственные imputers, encoder и scaler. Для MLP target StandardScaler также fit внутри training fold; `early_stopping=False`, поэтому нет случайного row validation, пересекающего семейства. Convergence warnings сохраняются в `training_warnings.csv`.

`cv_mae_mean/std` оценивают tuning каждого семейства по outer CV; выбор лучшего семейства из этих scores добавляет selection optimism. Это не unbiased оценка всей процедуры выбора семейства. `selection_cv_mae_mean/std` отдельно показывают winning search score на всем train. Семейство и его параметры зафиксированы до test; test metrics сохранены для всех трех моделей. Сравнение с исходным Midterm допускается только при одинаковых dataset/split hashes.

Run сохраняет сжатые полные pipelines в `models/endterm_v1/`, таблицы в `reports/tables/endterm_v1/`, PCA/KMeans и cluster assignments в `models/endterm_v1/segments/`. PCA fit только training X, k=2–6 выбирается по training silhouette без target. Engine/class/target cluster summaries вычисляются после выбора k и имеют описательный смысл. Existing run не перезаписывается. До сохранения валидного benchmark минимум 1 000 строк обычный запуск отвергается.

## Final: контролируемое сравнение текста

```powershell
python -m fuel_consumption.final --dataset data/processed/vehicles.parquet --split-dir data/splits --config configs/project.json --stage-config configs/final.json --run final_v1
python scripts/build_project_report.py --root . --runs midterm_v1 endterm_v1 final_v1 --output reports/project_v1
```

Четыре Ridge arms сравниваются на тех же IDs: structured; structured + model name; structured + engine description; оба text fields. Sanitizer удаляет leakage-bearing текст; TF-IDF fit внутри training folds. Alpha и arm выбираются только по train grouped CV. Эти CV scores являются **selection CV**, поскольку участвуют в выборе параметров и признаков; они не выдаются за unbiased estimate tuning. Test scores, per-row predictions и paired absolute-error differences сохранены; negative text-minus-structured delta означает меньшую ошибку text arm на этих test IDs.

Полный pipeline, stage config, checksum и training-only interface schema находятся в `models/final_v1/`; результаты — в `reports/tables/final_v1/`. Combined report проверяет исходные hashes и использует выполненные артефакты трех этапов без обучения. Один и тот же test после Midterm уже раскрыт; это сравнительный benchmark, не свежая независимая проверка дальнейшего исследования. Отдельные Endterm/Final course briefs не предоставлены, поэтому эти стадии реализуют исходную постановку демонстрационного проекта.

## Локальное приложение и повторный прогноз

```powershell
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 -- --run models/final_v1
```

Открыть локальный адрес, который напечатает Streamlit. Форма читает допустимые категории и defaults только из training-derived interface schema. Выбранный по train CV Final pipeline принимает семь specifications и доступный для своего arm исходный текст; optional engine text можно оставить пустым. Model designation также используется для проверки scope и обязателен в охваченных reviewed rules диапазонах Audi/Volvo, даже если не является prediction feature. Совместимые, но неполные engine conditions требуют уточнения. Приложение проверяет checksum до загрузки модели и не выполняет retraining. Для structured-only прогноза можно указать `--run models/midterm_v1`; Endterm run эта версия интерфейса не поддерживает.

Тот же проверенный pipeline доступен из Python:

```python
from fuel_consumption.predict import load_verified_model, predict_vehicle

bundle = load_verified_model("models/final_v1")
specifications = {
    "model_year": 2020,
    "manufacturer": "Honda",
    "displacement_l": 2.0,
    "cylinders": 4,
    "transmission": "Automatic (S6)",
    "drivetrain": "Front-Wheel Drive",
    "vehicle_class": "Compact Cars",
    "model_name": "Civic",
    "engine_description": "",
}
estimate_l100km = predict_vehicle(bundle, specifications, gasoline_nonhybrid=True)
print(estimate_l100km)
```

Это пример ввода спецификаций, не сохраненный прогноз конкретной EPA записи. При загрузке собственного project artifact wrapper проверяет saved selection, config, interface и model hashes; preprocessing и текстовое преобразование остаются внутри обученного pipeline. Возвращается EPA-estimated combined L/100 km, а не фактический расход конкретного водителя.

## Notebook, отчет и слайды

После завершенного run:

```powershell
python scripts/build_notebook.py --run-dir models/midterm_v1
python scripts/build_midterm_report.py --run-dir models/midterm_v1 --output-dir reports/midterm/midterm_v1
python scripts/prepare_slide_summary.py --dataset data/processed/vehicles.parquet --split-dir data/splits --tables reports/tables/midterm_v1 --snapshot data/raw/benchmark_2015_2025_20261009 --config configs/project.json --output reports/slides/slide_summary.json
```

Notebook в режиме `--run-dir` читает и проверяет сохраненные результаты без retraining. Без этого аргумента notebook заново обучает четыре модели во временной папке. Report generator сохраняет Markdown, четыре scientific figures и таблицы; не перезаписывает существующий report directory. После генерации выполнить notebook сверху вниз в kernel установленного проекта, сохранить все outputs.

`scripts/build_midterm_slides.mjs` создает editable PPTX из slide summary и использует bundled `@oai/artifact-tool` и finalizer Codex; это дополнительная среда для сборки слайдов, не зависимость collection/ML pipeline. Готовые слайды читаются в обычном PowerPoint. Восемь слайдов и speaker notes рассчитаны на 9 минут 20 секунд как демонстрационный сценарий; фактическая защита или репетиция не заявляются.

## Проверка опубликованного пакета

После установки среды из корня Git checkout:

```powershell
python scripts/verify_release.py --slides reports/slides/midterm_v1.pptx --slides-receipt reports/slides/midterm_v1.validation.json --slides-visual-review reports/slides/midterm_v1.visual_review.json --pytest-json evidence/offline_test_results.json --dry-run
```

Проверка сверяет raw response hashes, provenance, dataset/split, параметры и сохраненные результаты всех этапов, notebook, PPTX, ссылки и состав tracked files. Она не собирает данные и не обучает модели. `--pytest-json` проверяет evidence фактически выполненных 213 tests и неизменность исходников; новые tests этот скрипт не запускает. Без `--dry-run` после всех успешных проверок обновляется `evidence/PACKAGE_VERIFICATION.json`. Изменения tracked files сначала следует stage; ZIP без `.git` для этой проверки нужно распаковать в Git checkout либо проверить по его опубликованному commit.

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

### Исторический smoke config и последующие этапы

Исторический пример содержит 146 retained rows и отдельный frozen split. Его первоначальный config сохранен в `examples/smoke/models/first_20261009/config.json`; историческое поле status не переписывается задним числом. Root config уже изменен для основного сбора, поэтому он не совпадает с example checksum, даже если часть modeling settings одинакова. Для технического прогона на этом примере используются его исходный config и явная отметка `--allow-small`:

```powershell
python -m fuel_consumption.endterm --dataset examples/smoke/data/processed/vehicles.parquet --split-dir examples/smoke/data/splits --config examples/smoke/models/first_20261009/config.json --stage-config configs/endterm.json --run smoke_endterm --output-root work/private_stage_smoke --midterm-run examples/smoke/models/first_20261009 --allow-small
python -m fuel_consumption.final --dataset examples/smoke/data/processed/vehicles.parquet --split-dir examples/smoke/data/splits --config examples/smoke/models/first_20261009/config.json --stage-config configs/final.json --run smoke_final --output-root work/private_stage_smoke --allow-small
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 -- --run work/private_stage_smoke/models/smoke_final
```

Эти команды создают новые private development runs; повторный запуск с тем же run ID отвергается. Их scores не публикуются как accuracy достаточного основного benchmark. Dataset и split остаются исходными, а collection в этом блоке не запускается.
