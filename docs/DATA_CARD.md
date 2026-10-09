# Карточка данных первой реализации

Статус: **ограниченный реальный технический snapshot**, не полный Midterm dataset. Группа SE-2421: Tsybus Nikita, Bakytzhan Kassymgali. Minimum 1 000 и desired 3 000 строк еще не достигнуты.

Источник: [FuelEconomy.gov API](https://www.fueleconomy.gov/feg/ws/index.shtml). Одна строка — исходная EPA configuration с уникальным vehicle ID после проверок фильтрации и дубликатов. Это стандартизованная EPA fuel-economy оценка, не реальный расход владельца.

## Сбор

| Поле | Фактическое значение |
|---|---|
| Snapshot | smoke_20261009 |
| Метод | Собственный Python menu traversal и individual vehicle requests |
| Начало UTC | 2026-10-09T00:30:07.318989+00:00 |
| Окончание snapshot verification UTC | 2026-10-09T00:39:43.343168+00:00 |
| Timezone пользователя | Asia/Qyzylorda |
| Model years | 2015, 2020, 2025 |
| Производители | Ford, Honda, Toyota |
| Получено raw vehicle records | 250 |
| HTTP attempts/responses | 429 |
| Full catalogue complete | false |
| Raw metadata | examples/smoke/raw/requests.jsonl и snapshot.json |
| Clean dataset SHA-256 | f03e84f68e8e15daab31e6a1ce6e1e0886e5c33bac95a551ec5ecc6d19bfb665 |
| Split SHA-256 | 670cddb2508c011e750aed105eef6ba79c46ba144e8e91bd1067ab983098cb4a |

Пауза между запросами не менее 1 секунды, один worker. Ограничение объема использовано только для технического example. Полный collector сначала перечисляет все ID, затем получает отдельные записи.

## Очистка и поля

250 raw records → **146** accepted, **104** excluded. Подтвержденных дублей удалено 0; unresolved candidate groups 0. Все причины и последовательные before/after counts находятся в examples/smoke/data/interim/. Scope исключает hybrids/mild hybrids/alternative fuel и классы вне cars/SUV; metadata, numbers и target проверяются явно.

Target = 235.2145833333333 / comb08, US MPG → L/100 km. Target не импутируется, comb08U fallback отсутствует. X строго содержит семь predictors из config. Model name и доступный original engine description сохранены для Final; text в Midterm не используется.

Схема и назначение итоговых полей — [DATA_CONTRACT.md](DATA_CONTRACT.md) и examples/smoke/data/processed/schema.json. Машинный inventory всех raw keys/types/counts/disposition находится в examples/smoke/data/interim/raw_field_dictionary.csv. **Полный семантический словарь каждого raw field пока не завершен**; при подготовке окончательной data card нужен предметный review по официальной документации.

Пропуски engine description: 47 из 146. Original model names непустые. Статистическая импутация выполняется только внутри train pipelines.

## Split и ограничения

116 train / 30 test; 27 / 7 families, 5 grouped CV folds на train. Пересечения ID и groups = 0, unresolved baseModel fallbacks = 0. Сохранены mapping, assignments, coverage и hashes. Это split малого технического example, не финальный benchmark.

Ограничения: только три производителя и три модельных года в текущем example; малые подгруппы; EPA estimate и rounded MPG вместо наблюдаемого расхода; нет массы/мощности/аэродинамики среди начальных features. Accuracy conclusions для полного каталога пока не сформулированы.

## Источник и условия

Source access без логина подтвержден реальными HTTP responses. Сохранены API documentation и Privacy/Security policy в evidence/source_policy. [Политика](https://www.fueleconomy.gov/feg/ORNL-disclaimer.htm) предусматривает образовательное некоммерческое использование документов; проект не собирает фотографии или driver MPG. Автоматическое получение robots.txt в этой сессии завершилось закрытием соединения; пользователь ранее сообщил о проверке. Эта попытка не объявляется сохраненной проверенной копией robots.txt.

Следующая версия data card должна описать полный snapshot 2015–2025, достаточный объем, все поля, актуальные source checks, category audit и окончательный benchmark.
