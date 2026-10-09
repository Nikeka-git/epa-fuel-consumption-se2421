# Вклад команды и использование AI

Группа **SE-2421**. Участники сценария: **Tsybus Nikita** и **Bakytzhan Kassymgali**. Это демонстрация выполнения университетского проекта с помощью ИИ. Роли участников условны и заданы пользователем; фактическую реализацию, сбор, эксперименты, документацию и автоматические/визуальные проверки выполнил **OpenAI Codex** с параллельными AI-агентами. Человеческое авторство AI-generated code и ручная человеческая проверка не утверждаются.

## Условные роли в завершенном сценарии

| Участник | Завершенная часть сценария | Проверяемые артефакты |
| --- | --- | --- |
| Tsybus Nikita | Data/collection: постановка вопроса, собственный API collector, raw provenance и resume, gasoline/non-hybrid filtering, target conversion, powertrain/category/duplicate audit, data card и словарь | [raw snapshot](../data/raw/benchmark_2015_2025_20261009/snapshot.json), [cleaning summary](../data/interim/cleaned_summary.json), [независимый review](../reports/audit/benchmark_2015_2025_20261009/INDEPENDENT_REVIEW.md), [DATA_CARD](DATA_CARD.md); `api.py`, `collect.py`, `clean.py`, `powertrain.py` |
| Bakytzhan Kassymgali | Modeling/analysis/presentation/app: reviewed family grouping, frozen split/CV, pipelines, Midterm, tuned ensembles и MLP, PCA/KMeans, Final text ablations, ошибки, notebook, report, слайды и local inference | [split](../data/splits/split_metadata.json), [общий отчет](../reports/project_v1/RESULTS.md), [notebook](../notebooks/01_midterm.ipynb), [slides](../reports/slides/midterm_v1.pptx), [app](../app/streamlit_app.py); `split.py`, `features.py`, `train.py`, `endterm.py`, `segments.py`, `text.py`, `final.py`, `predict.py`, `evaluate.py` |

«Завершенная» означает, что соответствующий артефакт реально создан и проверен в демонстрации; это не запись личных ручных действий студентов. Оба участника в реальном курсе должны понимать вопрос, scope, формулу target, leakage, grouped split, preprocessing, CV, метрики, ошибки и ограничения всей работы. Их знания и реальные ответы здесь не оценивались.

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
| 2026-10-09 | Offline tests, notebook, отчеты, scientific PNG и презентация | [213 passed за 59.30 s](../evidence/offline_test_results.json); main notebook 12 code cells без errors; report четыре PNG; 8 slides с portable structural/font-policy evidence и отдельным PNG visual review |

AI assistance относится к исходному коду, решениям по данным, исследованиям первичных источников, запуску реального сбора/обучения, интерпретации результатов, QA и документации. Проверки выполнялись AI и программами; отсутствие отдельной человеческой валидации раскрывается явно. Если требуется точное имя модели для submission, его следует взять из фактических настроек сессии, а не угадывать по названию инструмента.

Исторический [smoke example](../examples/smoke/README.md) — 250 raw / 146 retained — сохранен отдельно. Его результаты и private development runs не выданы за основной benchmark. Main модели во всех этапах используют один dataset/split; train CV определяет выбор до test. Extra Trees test MAE 0.6900 не превращает его в выбранную модель после просмотра test; CV-selected tuning и text pipeline не объявляются улучшившими test без доказательства.

## Границы демонстрации

Источник описывает конфигурации автомобилей, не частных людей. API и Privacy/Security responses сохранены; неудачная автоматическая загрузка robots.txt не объявляется успешной проверкой, ранее пользователь сообщил о своей проверке. Blank EPA technology labels и focused manufacturer rules не дают исчерпывающей независимой сертификации non-hybrid powertrain. Семь raw encodings остаются unconfirmed. Native PowerPoint opening и native font rendering не проверялись; slide QA использовала portable checks и PNG, browser QA — один наблюдавшийся narrow viewport.

Регистрация у преподавателя, согласование неизвестных Endterm/Final briefs, pre-defense, репетиция по секундомеру, выступление, индивидуальные вопросы, оценка и Moodle submission **не выполнялись и N/A для AI demonstration**. Наличие готовых speaker notes не доказывает выступление обоих участников. Публичная публикация в [Nikeka-git/epa-fuel-consumption-se2421](https://github.com/Nikeka-git/epa-fuel-consumption-se2421) разрешена пользователем; состояние публикации отражает основная инструкция передачи.
