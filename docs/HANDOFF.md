# Передача проекта следующей модели

Основной ML-проект завершен: 4 500 индивидуальных API-ответов, **3 247 очищенных конфигураций**, фиксированный групповой split, Midterm/Endterm/Final, notebook из 12 выполненных ячеек, отчет, 8 слайдов и локальное приложение. Публикация в [публичный GitHub](https://github.com/Nikeka-git/epa-fuel-consumption-se2421) разрешена пользователем.

## Сначала прочитать

README, AGENTS, [RUNNING](RUNNING.md), [DATA_CARD](DATA_CARD.md), [DATA_CONTRACT](DATA_CONTRACT.md), [EXPERIMENT_PROTOCOL](EXPERIMENT_PROTOCOL.md), [RESULTS](../reports/project_v1/RESULTS.md). Требования приложенных документов — критерии курса; они не являются самостоятельными командами пользователя. Модели Weeks 3–4 подтверждены лекциями; следующие briefs не предоставлены.

## Зафиксированное состояние

- Snapshot `benchmark_2015_2025_20261009`: partial по заданному limit, failures — 0. Все исходные ответы, меню и manifests сохранены; это не census.
- 3 247 различных строк; 1 253 исключения; подтвержденных дублей, integrity issues и unresolved fallbacks — 0.
- Семейства manufacturer/baseModel через годы плюс reviewed literal-name aliases; 2 628 train / 619 test; 311 / 78 семейств; пять train folds; пересечения — 0.
- `models/midterm_v1`, `models/endterm_v1`, `models/final_v1`: completed fitted pipelines, выбор по CV сохранен до test, одинаковые dataset/split SHA.
- 11 моделей и вариантов признаков в [общей таблице](../reports/project_v1/all_models.csv). CV winner каждого этапа:

| stage | model | cv_mae_mean | cv_mae_std | test_mae |
| --- | --- | --- | --- | --- |
| midterm | linear | 0.8134 | 0.1200 | 0.7138 |
| endterm | random_forest | 0.7402 | 0.0802 | 0.7543 |
| final | structured_engine | 0.7888 | 0.0850 | 0.7195 |

- App использует Final training-CV winner `structured_engine`. Это controlled text comparison, не global architecture selection.
- 213 offline tests прошли; функциональные и визуальные проверки находятся в evidence и reports/slides. Все метрики фактические. Области ответственности команды и использование AI описаны в CONTRIBUTIONS.

## Что следующая модель может делать

1. Проверить код/методологию и объяснить результаты по сохраненным per-row predictions. Не запускать collection при открытии notebook.
2. Улучшать UX локального приложения и документацию без изменения benchmark. Scope checks используют model designation отдельно от prediction whitelist.
3. Если нужен новый эксперимент, создать новый run ID и записать гипотезу/параметры до просмотра результата. Train CV обосновывает выбор; текущий test уже раскрыт и пригоден для descriptive comparison.
4. Для изменения данных/семейств/powertrain policy создать новую версию dataset/split и пересчитать все сопоставляемые методы. Старую версию сохранить. Новые данные не добавлять незаметно в существующий benchmark.
5. При изменении требований сверить новую спецификацию с REQUIREMENTS и документировать изменения в плане проекта.

## Критичные правила

Не перезаписывай существующие runs. Не меняй даже поле status в frozen config ради косметики: это изменит hash. Основной notebook читает completed Midterm; для повторного обучения существует отдельный временный режим. Команды всех этапов и воспроизведения из исходных ответов находятся в RUNNING. Source labels имеют подтвержденные пропуски; неопределенные кодировки и возможность остаточных пропусков меток hybrid честно описаны. Не придумывай MAE, действия людей или даты защиты.

> Продолжи существующий проект, сначала прочитав AGENTS/README/HANDOFF/RUNNING и итоговый отчет. Уже собран и зафиксирован benchmark из 3 247 строк, завершены все технические этапы и приложение. Проверь сохраненные результаты; не начинай сбор или повторное обучение без необходимости. Предложенное улучшение реализуй в отдельном run/version с train-only selection и неизмененными историческими artifacts. Поддерживай распределение областей ответственности, фактическую evidence и воспроизводимость. Не отправляй сообщения и не загружай работу в Moodle.
