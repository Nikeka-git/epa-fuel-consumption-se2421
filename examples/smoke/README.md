# Первый реальный технический прогон

9 октября 2026 года. Собрано **250 individual vehicle records**, принято **146**, исключено **104**. Это ограниченный smoke snapshot, а не достаточный Midterm dataset. Порог 1 000 строк и желательная цель 3 000 пока не выполнены.

## Объем и происхождение

- Model years: 2015, 2020, 2025. Производители: Ford, Honda, Toyota.
- Порядок обхода заранее задан seed 42 и чередованием годов/марок; target не использовался для выбора записей.
- 429 HTTP attempts/responses сохранены; 166 model options menus обработаны.
- 253 IDs обнаружено, 250 получено, оставшиеся IDs/непройденные меню не выдаются за завершенный каталог.
- Сбор остановлен по лимиту 250 vehicle records. `full_catalogue_complete=false`.
- Повторный resume после лимита проверил hashes и сделал **0 новых сетевых запросов**.
- Фактические даты и URL находятся в `raw/requests.jsonl`; состояние — `raw/snapshot.json`.
- Исторический `raw/config.json` сохраняет конфигурацию collection. Изменение root config status на «implemented» отражает этап реализации; target, filters, features и параметры моделей не менялись.

## Данные и оценка

Подтвержденных полных дублей удалено: 0. Неразрешенных duplicate-candidate groups: 0. Inventory provenance проверен; причины исключения могут перекрываться и не суммируются как количество отдельных машин.

Grouped split: **116 train / 30 test**, 27 train families / 7 test families. Пересечения ID и семейств отсутствуют. На train сохранены 5 GroupKFold folds. Разделение этого example не является основным frozen benchmark проекта.

Все значения ниже в **L/100 km**, кроме R², который хранится в CSV. Это результаты технического прогона на небольшом ограниченном каталоге; они не подтверждают точность на всем рынке. CV spread — std по 5 folds, не доверительный интервал.

| Модель | CV MAE mean ± std | Test MAE | Test RMSE |
|---|---:|---:|---:|
| dummy | 2.1603 ± 0.7238 | 1.5308 | 1.7852 |
| linear | 0.8960 ± 0.2223 | 0.6652 | 0.9327 |
| knn | 1.0815 ± 0.2865 | 0.7543 | 0.9092 |
| tree | 1.3243 ± 0.3732 | 1.0067 | 1.1888 |

Модель **linear** выбрана по train CV до test prediction. Одинаковые IDs и folds использованы для всех четырех моделей. Таблицы, per-row predictions и grouped errors находятся в `reports/tables/first_20261009/`; fitted pipelines и hashes — в `models/first_20261009/`.

## Воспроизведение

Инструкция — [docs/RUNNING.md](../../docs/RUNNING.md). Очистка, split и обучение offline воспроизводятся из `raw/`. Notebook [01_midterm.ipynb](../../notebooks/01_midterm.ipynb) выполнен на этом dataset: **8 code cells без ошибок**; при повторном запуске он заново обучает модели во временной папке. Итоговый offline suite проекта также прошел: **91 тест** (47 cleaner, 36 API/collector, 8 modeling).

Для полноценного результата нужно собрать 2015–2025 шире, получить минимум 1 000 пригодных distinct configurations, проверить все категории/дубли и полный смысл raw fields, затем отдельно зафиксировать основной benchmark. Малый example сохраняется как проверка работоспособности.
