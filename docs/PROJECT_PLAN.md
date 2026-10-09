# План и структура завершенного проекта

Группа **SE-2421**. Участники демонстрационного сценария: **Tsybus Nikita** и **Bakytzhan Kassymgali**. Название: **Predicting EPA Combined Fuel Consumption from Vehicle Specifications**. Направление — 8. New-car catalogues; задача — регрессия.

На 9 октября 2026 года завершены собственный API-сбор, очистка и независимый аудит, фиксированный grouped benchmark, Midterm, Endterm, Final text experiments и локальное приложение. Это демонстрация выполнения задания с помощью Codex: роли участников условные, исходные ответы API и результаты экспериментов реальные. Регистрация, выступление, pre-defense и Moodle submission не выполнялись; для демонстрации они N/A.

## Вопрос и границы данных

Исследуем, насколько точно можно предсказать EPA-estimated combined fuel consumption в **L/100 km** по объему двигателя, числу цилиндров, коробке передач, приводу, классу, производителю и модельному году. Результат помогает сравнивать паспортные оценки расхода; он не обещает фактический расход при конкретной погоде, маршруте или стиле вождения.

Источник — индивидуальные записи [FuelEconomy.gov API](https://www.fueleconomy.gov/feg/ws/index.shtml), рынок США, модельные годы **2015–2025**. Scope включает бензиновые легковые автомобили, универсалы и SUV. Дизельные, альтернативные и двухтопливные версии, гибриды, включая mild hybrids, и неподходящие классы исключаются по [контракту](DATA_CONTRACT.md). Название направления относится к каталогу исходных конфигураций, а не к утверждению, что модели 2015 года сейчас продаются новыми.

Собрано **4 500 raw vehicle records**; после **1 253 исключений** сохранено **3 247 различных пригодных конфигураций**, **49 производителей**, **12 классов**, все 11 лет. Удаленных подтвержденных дублей и нерешенных duplicate candidates нет. Минимум курса/пользователя 1 000 и желательная цель пользователя 3 000 достигнуты.

Сбор ограничен бюджетом и выполнялся seeded round-robin по всем запрошенным годам и доступным производителям. Первоначальный бюджет 6 000 уменьшен до 4 500 до split и основных fits; основание и фактические invocations сохранены в [collection provenance](../evidence/benchmark_collection_provenance.json). Snapshot имеет статус `partial`: это выборка с неодинаковыми вероятностями включения, не полный census и не распределение продаж. Семантически сомнительные SQ7/SQ8 явно quarantined, а пустые EPA technology labels не объявляются независимым доказательством отсутствия hybrid. Полный [предметный review](../reports/audit/benchmark_2015_2025_20261009/INDEPENDENT_REVIEW.md) сохраняет эти ограничения.

## Завершенные этапы

| Этап | Выполненная работа | Проверяемый результат |
| --- | --- | --- |
| Требования | Guide, Midterm Rubric и лекции Weeks 3–4; различение требований курса, пользователя и решений проекта | [REQUIREMENTS.md](REQUIREMENTS.md); подтверждены Linear, KNN и Regression Tree |
| Сбор | Собственный Python collector; individual requests, raw bytes, даты, hashes, retry/resume | [raw snapshot](../data/raw/benchmark_2015_2025_20261009/snapshot.json); 4 500 records, ноль failed attempts |
| Очистка и аудит | Scope, типы, target, категории, powertrain review и target-free duplicate audit | [cleaned summary](../data/interim/cleaned_summary.json); 3 247 rows; [95-field dictionary](../data/interim/raw_field_dictionary.json) |
| Freeze | Проверены 13 connected source-alias components; сохранены семейства, train/test и пять train folds | [split metadata](../data/splits/split_metadata.json); 389 groups, нулевые ID/group overlaps |
| Midterm | Train EDA, Dummy/Linear/KNN/Tree, grouped CV, test и error analysis | [исполненный notebook](../notebooks/01_midterm.ipynb), [report](../reports/midterm/midterm_v1/report.md), четыре PNG и восемь слайдов |
| Endterm | Nested grouped ensemble tuning, MLP, train-fitted PCA/KMeans | [metrics](../reports/tables/endterm_v1/metrics.csv); сохраненные модели и segments |
| Final | Четыре контролируемых Ridge/TF-IDF arms, paired test comparison, приложение | [metrics](../reports/tables/final_v1/metrics.csv); [Streamlit app](../app/streamlit_app.py) |
| Итоговая передача | Общий отчет, данные, pipelines, evidence, инструкции и условные роли | [общий отчет](../reports/project_v1/RESULTS.md), [HANDOFF.md](HANDOFF.md), [CONTRIBUTIONS.md](CONTRIBUTIONS.md) |

Отдельные Endterm/Final briefs и полный syllabus Weeks 5–8 не предоставлены. Эти этапы реализуют программу Guide и предложение пользователя; соответствие неизвестной отдельной рубрике не утверждается. В реальном курсе Midterm приходится на Week 5, Endterm на Week 9, обязательный неоцениваемый pre-defense на Week 10, Final — после Week 10. Конкретная календарная дата защиты неизвестна.

## Зафиксированный протокол

Target: `235.2145833333333 / comb08`, без imputation или построчной подмены на `comb08U`. Structured X содержит ровно семь указанных технических признаков; fuel economy, fuel costs, emissions, efficiency scores, ID, даты и grouping fields не входят в X.

Одна неизменная выборка и один split используются во всех трех этапах. Group definition — нормализованные manufacturer/baseModel с проверенным source alias map, через годы. После alias review **389 групп**: **2 628 train rows / 311 groups**, **619 test rows / 78 groups**. GroupShuffleSplit: seed 42, 20% групп; на train — пять GroupKFold folds. Препроцессинг и подбор обучаются только внутри соответствующих training folds. Полный [протокол](EXPERIMENT_PROTOCOL.md) и manifests задают порядок проверки.

Модели выбираются по training CV до test prediction. Midterm CV оценивает фиксированные модели; Endterm outer CV — inner-search procedure каждого семейства, но последующий выбор семейства добавляет optimism. Final CV выбирает alpha и text arm и является selection CV. Разные роли CV не выдаются за одинаковые независимые оценки. Test повторно используется для сравнений этапов, а не для изменения параметров, seed, aliases или scope.

## Фактические результаты

| Этап | Выбрано по CV | CV MAE | Test MAE, L/100 km | Вывод |
| --- | --- | ---: | ---: | --- |
| Midterm | Linear Regression | 0.8134 | 0.7138 | Dummy test MAE 2.5970; прогноз полезнее константы |
| Endterm | Random Forest | 0.7402 | 0.7543 | Выбранный ансамбль не улучшил test MAE выбранной Midterm модели |
| Final | Structured + engine text, Ridge | 0.7888 | 0.7195 | Выбранный text arm хуже paired structured Ridge: 0.7046 |

Extra Trees имеет test MAE **0.6900**, но не был CV winner; MLP — **0.7581**, 164 iterations, ноль зарегистрированных training/convergence warnings. Эти test числа не использованы для замены выбранного алгоритма. На Final model-name arm дает описательный test MAE **0.6966** и оба текста **0.7006**; это также не основание выбрать их после просмотра test. Улучшение всех ансамблей или всех текстовых вариантов не заявляется.

PCA обучена на train structured representation с масштабированными числами и one-hot категориями: **49 components** для заданного порога retained variance. KMeans выбрал **k=2**, training silhouette **0.2544**, без target в fit/selection. Кластеры — описательный анализ структуры, не доказательство качества регрессии или причинных сегментов.

## Архитектура и структура файлов

Поток: **API → immutable raw и provenance → scope/cleaning/audit → frozen dataset/split → train CV → test tables → saved pipeline → local app**. Offline notebook читает сохраненные main результаты; при его выполнении сетевой сбор и повторный fit не запускаются.

| Компонент | Ответственность | Выход |
| --- | --- | --- |
| `api.py`, `collect.py` | Bounded HTTP workers, глобальный rate interval, retry/Retry-After, single writer, atomic cache и resume | `data/raw/<snapshot>/` |
| `clean.py`, `powertrain.py` | Типы, фильтры, target, supplementary rules, полный audit | `data/interim/`, `data/processed/` |
| `split.py` | Проверенные aliases, group mapping, train/test, folds и hashes | `data/splits/` |
| `features.py`, `train.py`, `evaluate.py` | Strict structured allowlist, fold-local pipelines, Midterm и error tables | `models/midterm_v1/`, `reports/tables/midterm_v1/` |
| `endterm.py`, `segments.py` | Nested grouped tuning, MLP, PCA и KMeans | `models/endterm_v1/`, Endterm tables |
| `text.py`, `final.py` | Target-neutral sanitizer, fixed word 1–2 TF-IDF budget, Ridge ablations | `models/final_v1/`, Final tables |
| `predict.py`, `app/streamlit_app.py` | Hash-checked fitted model, input validation и согласованный powertrain scope | Local prediction в L/100 km |
| `scripts/`, `notebooks/`, `reports/` | Проверяемые отчеты и презентация из сохраненных artifacts | Notebook, CSV/JSON, PNG, PPTX и Markdown |

## Проверки и ограничения передачи

Полный offline suite: **213 passed за 59.30 s**, [запись реального выполнения и source hashes](../evidence/offline_test_results.json). Main notebook выполнен: **12 code cells, ноль errors**; report содержит четыре PNG. Слайды имеют восемь страниц, speaker notes, portable package/layout/font-policy evidence и [индивидуальный PNG review](../reports/slides/midterm_v1.visual_review.json). Native PowerPoint opening и native font rendering не проверялись. Timing в notes — план, не замеренная репетиция.

Main app функционально проверено [Streamlit AppTest](../evidence/app_functional_qa.json); отдельный [browser review](../evidence/app_browser_visual_qa.json) проверил читаемость/scrolling при viewport 525×530. Приложение загружает CV-selected `structured_engine`, принимает семь specifications и доступный текст, проверяет scope и возвращает L/100 km. Оно не переобучает модель при запуске и не заявляется глобально лучшей архитектурой.

Из 95 raw fields семь encoding meanings остаются честно unconfirmed и исключены из predictors. Все семь X заполнены; engine description отсутствует у **777** строк, которые сохраняются в text comparison. Rounded source MPG, неполные powertrain labels, неравное покрытие и редкие группы ограничивают выводы. Роли людей условны; фактическую автоматизацию и проверки выполнил Codex. Исходный DOCX изучен по OOXML text/tables, но visual rendering не выполнен из-за отсутствия bundled LibreOffice. Сохранены source-policy responses; автоматическая загрузка robots.txt не удалась, предшествующая проверка заявлена пользователем.

## Продолжение работы

Следующая модель может воспроизвести pipeline offline по [RUNNING.md](RUNNING.md), проверить [общий отчет](../reports/project_v1/RESULTS.md) и изучить subgroup errors. Для нового эксперимента нужна отдельная версия/run с заранее заданным train-only protocol; нельзя перезаписывать frozen benchmark или подбирать решения по уже опубликованному test. Новый доступный course brief следует сопоставить с [матрицей требований](REQUIREMENTS.md).

В демонстрационном распределении Nikita завершает data/collection часть, Bakytzhan — modeling, analysis, presentation и app. Это назначение ролей, а не утверждение личного авторства AI-generated code. Реальная регистрация, live defense, индивидуальные ответы и Moodle upload остаются N/A для этого сценария.
