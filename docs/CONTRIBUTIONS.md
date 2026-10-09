# Вклад команды и использование AI

Группа **SE-2421**. Team members: **Tsybus Nikita** и **Bakytzhan Kassymgali**. Проект охватывает полный ML workflow: собственный сбор данных, аудит, моделирование, анализ и приложение. Для реализации кода, запуска сбора и экспериментов, документации и автоматических/визуальных проверок использовался **OpenAI Codex** с параллельными AI-агентами. Ниже разделены области ответственности команды и конкретное использование AI.

## Области ответственности команды

| Team member | Область проекта | Проверяемые артефакты |
| --- | --- | --- |
| Tsybus Nikita | Data/collection: постановка вопроса, собственный API collector, raw provenance и resume, gasoline/non-hybrid filtering, target conversion, powertrain/category/duplicate audit, data card и словарь | [raw snapshot](../data/raw/benchmark_2015_2025_20261009/snapshot.json), [cleaning summary](../data/interim/cleaned_summary.json), [независимый review](../reports/audit/benchmark_2015_2025_20261009/INDEPENDENT_REVIEW.md), [DATA_CARD](DATA_CARD.md); `api.py`, `collect.py`, `clean.py`, `powertrain.py` |
| Bakytzhan Kassymgali | Modeling/analysis/presentation/app: reviewed family grouping, frozen split/CV, pipelines, Midterm, tuned ensembles и MLP, PCA/KMeans, Final text ablations, ошибки, notebook, report, слайды и local inference | [split](../data/splits/split_metadata.json), [общий отчет](../reports/project_v1/RESULTS.md), [notebook](../notebooks/01_midterm.ipynb), [slides](../reports/slides/midterm_v1.pptx), [app](../app/streamlit_app.py); `split.py`, `features.py`, `train.py`, `endterm.py`, `segments.py`, `text.py`, `final.py`, `predict.py`, `evaluate.py` |

Таблица связывает области ответственности с компонентами и артефактами проекта. Общая методологическая основа — вопрос исследования, scope, target conversion, leakage controls, grouped split, fold-local preprocessing, CV, метрики и анализ ошибок. Инструменты и задачи автоматизации зафиксированы в отдельном AI log.

## Фактически выполненная Codex-автоматизация

| Дата | Работа AI | Реальная проверка или результат |
| --- | --- | --- |
| 2026-10-09 | Чтение Guide, Midterm Rubric и лекций Weeks 3–4; план, контракт, protocol и pilot | PDF text/visual review; DOCX OOXML text и все четыре таблицы. Visual DOCX rendering недоступен из-за отсутствия bundled LibreOffice |
| 2026-10-09 | Реализация и запуск собственного API-сбора; bounded workers/global interval, immutable raw/hash/manifest, retry и resume | 4 500 raw records, 8 568 successful recorded requests, ноль failed attempts; [collection provenance](../evidence/benchmark_collection_provenance.json) |
| 2026-10-09 | Очистка, documented field dictionary, предметный powertrain review и независимый scope/dedup audit | 3 247 retained, 1 253 excluded, 49 makes, 12 classes; 95 raw fields, семь unconfirmed encodings исключены из X. Source/target/feature integrity discrepancies — ноль |
| 2026-10-09 | Identity-only family review и freeze | 13 reviewed components; 389 groups; 2 628 train / 619 test rows; 311/78 groups; пять reused training folds; нулевые ID/group overlaps |
| 2026-10-09 | Midterm: Dummy, Linear, KNN и Tree; train EDA, CV, test и error reports | CV выбрала Linear; test MAE 0.7138 против Dummy 2.5970 L/100 km. [Реальные tables](../reports/tables/midterm_v1/metrics.csv) |
| 2026-10-09 | Endterm: nested grouped tuning Random Forest/Extra Trees, MLP; PCA/KMeans | CV выбрала RF, test MAE 0.7543; MLP 0.7581, 164 iterations, ноль warnings. PCA 49 components, KMeans k=2, train silhouette 0.2544; target не использован для cluster selection |
| 2026-10-09 | Final: четыре Ridge/text arms, target-neutral sanitizer, fixed TF-IDF budget и paired comparison | CV выбрала structured + engine text: test MAE 0.7195; structured Ridge 0.7046. Выбранный текст не улучшил held-out MAE; альтернативные test scores не изменили CV выбор |
| 2026-10-09 | Сохраненные pipelines и Streamlit app; согласованная проверка specifications/powertrain scope | [Main AppTest](../evidence/app_functional_qa.json) и [browser QA](../evidence/app_browser_visual_qa.json) при viewport 525×530; ошибка scope корректно отклоняется, допустимый ввод возвращает finite positive prediction |
| 2026-10-09 | Offline tests, notebook, отчеты, scientific PNG и презентация | [213 passed за 51.19 s](../evidence/offline_test_results.json); main notebook 12 code cells без errors; report четыре PNG; 8 slides с portable structural/font-policy evidence и отдельным PNG visual review |
| 2026-10-09 | Редакционное оформление документации, отчета и презентации | [Редакционная provenance](../evidence/editorial_revision.json); исходные training hashes сохранены, параметры, данные, split и fitted models не изменены; повторный offline suite и AppTest прошли |

AI assistance относится к исходному коду, решениям по данным, исследованиям первичных источников, запуску реального сбора/обучения, интерпретации результатов, QA и документации. Проверки зафиксированы в automated test receipts, app QA, source review и visual review evidence. Если требуется точное имя модели для submission, его следует взять из фактических настроек сессии, а не угадывать по названию инструмента.

Исторический [smoke example](../examples/smoke/README.md) — 250 raw / 146 retained — сохранен отдельно. Его результаты и private development runs не выданы за основной benchmark. Main модели во всех этапах используют один dataset/split; train CV определяет выбор до test. Extra Trees test MAE 0.6900 не превращает его в выбранную модель после просмотра test; CV-selected tuning и text pipeline не объявляются улучшившими test без доказательства.

## Ограничения источников и проверки

Источник описывает конфигурации автомобилей, не частных людей. API и Privacy/Security responses сохранены; неудачная автоматическая загрузка robots.txt не объявляется успешной проверкой, ранее пользователь сообщил о своей проверке. Blank EPA technology labels и focused manufacturer rules не дают исчерпывающей независимой сертификации non-hybrid powertrain. Семь raw encodings остаются unconfirmed. Native PowerPoint opening и native font rendering не проверялись; slide QA использовала portable checks и PNG, browser QA — один наблюдавшийся narrow viewport.

Код, данные, воспроизводимые эксперименты и приложение опубликованы в [Nikeka-git/epa-fuel-consumption-se2421](https://github.com/Nikeka-git/epa-fuel-consumption-se2421). Guide, Midterm Rubric и лекции служат источниками учебных критериев; отдельные Endterm/Final briefs и полный syllabus следующих недель не предоставлены. Проектные решения и технические ограничения изложены в [REQUIREMENTS.md](REQUIREMENTS.md) и [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).
