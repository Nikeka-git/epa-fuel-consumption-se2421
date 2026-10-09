# Predicting EPA Combined Fuel Consumption

Воспроизводимый ML-проект по прогнозированию расхода топлива автомобилей. Группа **SE-2421**, команда **Tsybus Nikita** и **Bakytzhan Kassymgali**; направление **8. New-car catalogues**. Области ответственности и использование AI описаны в [CONTRIBUTIONS](docs/CONTRIBUTIONS.md).

Вопрос: насколько точно технические характеристики позволяют предсказать EPA combined fuel consumption бензиновых автомобилей, универсалов и SUV рынка США за model years 2015–2025? Target — **L/100 km**. Источник — [FuelEconomy.gov API](https://www.fueleconomy.gov/feg/ws/index.shtml).

Публичный репозиторий: [Nikeka-git/epa-fuel-consumption-se2421](https://github.com/Nikeka-git/epa-fuel-consumption-se2421).

## Выполнено

Собственным Python-кодом получено **4500 индивидуальных API-записей**. После фильтрации сохранено **3247 различных конфигураций**, **49 производителей**, все 11 модельных лет. Raw bytes, menus, даты, manifests и SHA-256 включены. Это bounded sample, а не полный каталог или выборка продаж. Исключено 1253 записей; подтвержденных дублей и проблем целостности — 0.

Выполнены Midterm (Dummy, Linear, KNN, Decision Tree), Endterm (tuned Random Forest, Extra Trees, MLP; PCA/KMeans), Final (четыре Ridge/TF-IDF ablations) и локальное приложение. Для всех этапов используются одни данные, split и пять grouped folds. **2628 train / 619 test; 311 / 78 source families**. Полный offline suite: **213 passed**.

## Результаты

Победитель каждого этапа выбран по training CV до test prediction. Все MAE в L/100 km:

| stage | model | cv_mae_mean | cv_mae_std | test_mae |
| --- | --- | --- | --- | --- |
| midterm | linear | 0.8134 | 0.1200 | 0.7138 |
| endterm | random_forest | 0.7402 | 0.0802 | 0.7543 |
| final | structured_engine | 0.7888 | 0.0850 | 0.7195 |

Midterm CV использует фиксированные параметры, Endterm — nested grouped CV процедуры tuning для каждого семейства моделей, Final — selection CV для alpha и набора текста. Эти CV-оценки имеют разные роли; fold std не является доверительным интервалом. Подробные результаты всех 11 вариантов, эффект текста, ошибки, предупреждения и ограничения — [RESULTS](reports/project_v1/RESULTS.md).

## Открыть результаты

- [Исполненный offline notebook](notebooks/01_midterm.ipynb): 12 code cells, минимум три EDA-графика, ошибки и выводы.
- [Midterm report](reports/midterm/midterm_v1/report.md) и [8 слайдов с заметками](reports/slides/midterm_v1.pptx).
- [Итоговый отчет](reports/project_v1/RESULTS.md) и [полная таблица моделей](reports/project_v1/all_models.csv).
- [Карточка данных](docs/DATA_CARD.md), [протокол](docs/EXPERIMENT_PROTOCOL.md), [план и структура](docs/PROJECT_PLAN.md), [передача следующей модели](docs/HANDOFF.md).

## Запуск

Из корня репозитория:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock.txt
.venv/Scripts/python.exe -m pip install -e . --no-deps
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
```

На Linux/macOS использовать `.venv/bin/python`. Приложение загружает Final CV winner **structured_engine**, принимает характеристики и доступный текст и выводит L/100 km. Это выбранный вариант контролируемого Final-сравнения, не автоматически лучшая архитектура всех этапов. Оно не обучает модель при открытии. Подробные команды воспроизведения — [RUNNING](docs/RUNNING.md).

## Основные решения

- `target_l100km = 235.2145833333333 / comb08`, где `comb08` — US MPG; target не импутируется.
- Structured X содержит только `model_year`, `displacement_l`, `cylinders`, `manufacturer`, `transmission`, `drivetrain`, `vehicle_class`.
- Другие измерения экономичности, стоимость топлива, emissions, scores и identifiers не входят в X. Original text сохраняется, но проходит target-neutral sanitizer внутри Final pipeline.
- Семейства группируются через годы по manufacturer/baseModel с явно reviewed aliases. Это независимость заявленных source families, не всех возможных поколений или платформ.
- Все imputers, scalers, encoders, TF-IDF, PCA и estimators fit только на train соответствующего fold. Test не участвует в tuning.
- EPA technology labels имеют подтвержденные пропуски; supplemental manufacturer review и uncertainty quarantine заданы до freeze. Это ограничение источника, не независимая сертификация каждой машины.

## Структура

```text
configs/                       frozen settings and reviewed powertrain rules
src/fuel_consumption/          collect, clean, split, train, endterm, final, predict
data/raw/                     immutable individual API responses and menus
data/interim/                 cleaning evidence and field dictionary
data/processed/               CSV, Parquet and schema
data/splits/                  fixed IDs, families, CV folds and hashes
models/                       saved fitted pipelines for all three stages
reports/                      audit, tables, scientific figures, slides, final report
notebooks/01_midterm.ipynb     executed offline analysis
app/streamlit_app.py           local prediction interface
scripts/                      reproducible report, dictionary and slide builders
docs/                         plan, contract, protocol, data card and handoff
tests/                        meaningful offline invariant checks
examples/smoke/               historical development example, separate split
references/                   supplied course guide, rubric and lectures
```

Проект включает полный цикл от сбора исходных данных до сохраненных моделей и локального приложения. Исходные требования и их связь с артефактами находятся в [REQUIREMENTS](docs/REQUIREMENTS.md).
