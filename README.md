# Predicting EPA Combined Fuel Consumption

Проект группы **SE-2421**, **Tsybus Nikita** и **Bakytzhan Kassymgali**. Направление: **8. New-car catalogues**. Задача: регрессия, целевая переменная — EPA combined fuel consumption в **L/100 km**.

Оцениваем, насколько точно технические характеристики позволяют предсказать паспортный расход бензиновых легковых автомобилей и SUV рынка США модельных лет 2015–2025. Данные собираются собственным Python-кодом через [FuelEconomy.gov API](https://www.fueleconomy.gov/feg/ws/index.shtml), с запросом отдельных записей автомобилей. Гибриды, включая mild hybrid, и альтернативные виды топлива исключаются.

Публичный репозиторий проекта: [Nikeka-git/epa-fuel-consumption-se2421](https://github.com/Nikeka-git/epa-fuel-consumption-se2421).

## Текущее состояние

Реализована первая рабочая версия: API-клиент и resumable collector, очистка и audit reports, строгий выбор семи признаков, групповой split и пять CV folds, обучение Dummy/Linear/KNN/Tree, сохранение pipelines, predictions и таблиц ошибок. Проверки прошли без ошибок: **91 offline тест** (47 для очистки, 36 для API/collector, 8 для modeling). Команды доступны в [инструкции запуска](docs/RUNNING.md).

В ограниченном реальном сборе получено **250 raw records**, из них принято **146**, исключено **104**. Scope и результаты описаны в [examples/smoke/README.md](examples/smoke/README.md). Raw responses, очищенные данные, отдельный split, модели и отчеты этого примера находятся в `examples/smoke/`. Он предназначен для проверки воспроизводимости; минимальный объем Midterm — **1 000 различных пригодных конфигураций** — еще предстоит получить. Цель основного benchmark — **3 000 и более** до первого freeze.

Полный Midterm, слайды, Endterm-модели, Final text experiments и локальное приложение еще не завершены. Ограниченный пример не подтверждает точность модели на всем каталоге 2015–2025.

## Запуск

Все команды выполняются из корня репозитория:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock.txt
.venv/Scripts/python.exe -m pip install -e . --no-deps
.venv/Scripts/python.exe -m pytest -q
```

На Linux/macOS использовать `.venv/bin/python`. Полный сбор, resume, обработка benchmark и offline воспроизведение smoke описаны в [docs/RUNNING.md](docs/RUNNING.md).

Notebook `notebooks/01_midterm.ipynb` создается через `scripts/build_notebook.py` и выполнен на smoke dataset: все **8 code cells** завершились без ошибок. Он работает offline на заданных dataset/split и обучает модели во временной папке; сетевой сбор при Run All не запускается. Перед полноценным Midterm нужно пересоздать notebook для основного benchmark и выполнить Restart and Run All.

## Основные решения

- `target_l100km = 235.2145833333333 / combined_mpg`, где исходный API field — `comb08`, US MPG. Target не импутируется.
- Midterm X: `model_year`, `displacement_l`, `cylinders`, `manufacturer`, `transmission`, `drivetrain`, `vehicle_class`. Остальные поля исключены из inputs по умолчанию.
- `model_name` и `engine_description` сохраняются для Final; `base_model` используется для группировки.
- Семейства `manufacturer + base_model` удерживаются целиком через годы. Fallback полного model name и возможные aliases отражаются в отдельном audit.
- `GroupShuffleSplit`: seed 42, test — 20% групп. На train — пять `GroupKFold` folds. Сохраненный manifest переиспользуется; изменение dataset/config не перезаписывает benchmark.
- Preprocessing обучается внутри каждого training fold. Модель выбирается по train CV MAE до test prediction; затем публикуются test scores всех четырех моделей.
- Главная метрика — MAE в L/100 km; дополнительно RMSE и R². Ошибки разбираются по vehicle class и размеру двигателя с числом наблюдений.

## Структура и документация

```text
configs/project.json             настройки сбора и эксперимента
src/fuel_consumption/            api, collect, clean, split, features, train, evaluate
tests/                          реальные API fixtures и синтетические проверки инвариантов
examples/smoke/                  отдельный ограниченный реальный пример
data/                           рабочие raw, interim, processed, splits
models/                         полные fitted pipelines отдельных runs
reports/                        таблицы и графики
notebooks/01_midterm.ipynb       offline исследование выбранного dataset
scripts/build_notebook.py        генератор воспроизводимого notebook
references/                     предоставленные Guide, Rubric и лекции
evidence/api_probe/              историческая техническая проверка пяти API records
docs/                           контракт, protocol, требования, план и журнал работы
```

В больших рабочих snapshots данные хранятся локально; `.gitignore` не заменяет их сохранение для сдачи. Исторические raw responses и опубликованный example должны сохраняться с manifests и SHA-256.

Прочитайте [план](docs/PROJECT_PLAN.md), [контракт данных](docs/DATA_CONTRACT.md), [протокол экспериментов](docs/EXPERIMENT_PROTOCOL.md) и [инструкцию продолжения](docs/HANDOFF.md). [Матрица требований](docs/REQUIREMENTS.md) отделяет требования курса от проектных решений; [CONTRIBUTIONS](docs/CONTRIBUTIONS.md) фиксирует фактическое использование AI и вклад участников.
