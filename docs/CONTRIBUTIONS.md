# Вклад команды и использование AI

Группа SE-2421. Участники сценария: Tsybus Nikita и Bakytzhan Kassymgali. Пользователь уточнил 9 октября 2026 года, что это **демонстрация выполнения университетского задания с помощью ИИ**, а не реальная сдача. Распределение работы между участниками ниже условное и задано по его запросу. Реальные API responses, даты, размеры данных, проверки и результаты моделей подтверждаются сохраненными artifacts. Фактически автоматизированную работу выполняет Codex, что отдельно отражено в AI log.

## Распределение выполненной работы в демонстрационном сценарии

| Участник сценария | Закрепленная часть работы | Файлы или commits | Дата |
|---|---|---|---|
| Tsybus Nikita | Постановка regression question; код собственного API collector, raw provenance, фильтрация gasoline/non-hybrid cars/SUV, target conversion, data card и словарь полей | configs/project.json; src/fuel_consumption/api.py, collect.py, clean.py; scripts/audit_benchmark.py; docs/DATA_CONTRACT.md и DATA_CARD.md | 2026-10-09 |
| Bakytzhan Kassymgali | Grouped split/CV, preprocessing pipelines, Dummy/Linear/KNN/Tree, evaluation и error analysis; воспроизводимый notebook, report и слайды | src/fuel_consumption/split.py, features.py, train.py, evaluate.py; notebooks/; reports/; scripts/build_notebook.py, build_midterm_report.py и build_midterm_slides.mjs | 2026-10-09 |

Объем реально завершенных artifacts отражается в README и HANDOFF. Эта таблица описывает роли демонстрационной команды и не утверждает, что люди лично написали AI-generated code или что еще не выполненный этап уже завершен.

## AI tools

| Дата | Инструмент | Использование | Проверка человеком |
|---|---|---|---|
| 2026-10-09 | OpenAI Codex, с параллельными AI-проверками | Чтение требований и лекций, план проекта, схема данных, protocol, pilot API script и техническая проверка пяти записей | Участникам предстоит проверить решения и объяснять итоговый код |
| 2026-10-09 | OpenAI Codex, с параллельными AI-агентами | Реализация api/collect/clean: parsing, immutable cache, retries, resume, filters, target conversion и audit reports; реальные API fixtures и offline тесты | AI выполнил automated checks; отдельная проверка человеком пока не зафиксирована |
| 2026-10-09 | OpenAI Codex, с параллельными AI-агентами | Реализация split/features/train/evaluate: grouped manifests, strict allowlist, train-only preprocessing, CV selection, pipelines и error tables; tests/test_modeling.py | 8 modeling tests прошли; итоговый offline suite — 91 тест: 47 cleaner, 36 API/collector, 8 modeling. Участникам предстоит проверить методологию и код |
| 2026-10-09 | OpenAI Codex | Запуск ограниченного реального collection, подготовка и выполнение offline notebook, фиксация зависимостей, технического example, документации и пакета для разрешенной публичной публикации на GitHub | 250 raw records, 146 accepted, 104 excluded; 8 code cells выполнены без ошибок. Scope/artifacts/results — examples/smoke/README.md; достаточный Midterm benchmark еще не завершен |

Если точное имя используемой модели требуется в submission, переписать его из фактических настроек сессии; не угадывать. При дальнейшей AI-разработке записывать инструмент, задачи, измененные файлы и реальную валидацию. Исторический pilot написан с помощью AI; команда должна понимать, проверить и документировать собственный воспроизводимый сбор.

Оба участника отвечают за понимание вопроса, scope, target conversion, leakage, split, preprocessing, CV, метрик, анализа ошибок и ограничений, независимо от распределения задач.

Публикация в публичный `Nikeka-git/epa-fuel-consumption-se2421` разрешена пользователем. Для этой демонстрации не требуется дополнять таблицу вымышленными ручными действиями: условное распределение ролей задано выше, фактическая автоматизация отражена в AI log. Организационные действия реального курса, защита и отправка в Moodle не входят в выполненную демонстрацию.
