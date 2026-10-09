# Протокол экспериментов

Первоначальный протокол 0.1, дополненный состоянием завершенного benchmark. Основная оценка показывает перенос на **группы manufacturer/baseModel с проверенными aliases, отсутствовавшие в обучении**, в каталоге 2015–2025. Эти группы следуют таксономии источника и не подтверждают физические поколения или платформы автомобилей. Оценка не проверяет будущие годы или новые рынки. Группировка — наше методологическое решение; курс требует фиксированный split и k-fold CV, но не предписывает конкретный splitter.

Выполнены все три этапа на 3 247 строках и одном frozen split: 2 628 train / 619 test, пять train folds. Фактические параметры, результаты 11 моделей и ограничения приведены в [итоговом отчете](../reports/project_v1/RESULTS.md). Ниже сохранены правила, заданные до оценки; необязательные предложения не означают, что соответствующие эксперименты проводились.

## Разделение данных

1. Завершить scope filtering, проверку targets и duplicate audit по заранее заданным правилам.
2. Определить семейство: нормализованный `manufacturer + base_model`; если base_model отсутствует, использовать полное model_name. Trim whitespace, унифицировать регистр и пробелы. Сохранять исходные значения. Год не включать: одно семейство через годы должно быть в одной группе.
3. До чтения target для выбора групп проверить aliases и случаи, где fallback model может совпадать с другим baseModel. Сохранить `group_mapping.csv` и source='base_model'/'fallback'/'audited_alias'. Не сокращать названия эвристически до первого слова. Fallback без проверки не доказывает полное отсутствие связанных семейств между splits.
4. Отсортировать по vehicle_id и выполнить `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`. Здесь 20% относится к количеству **групп**, доля строк может отличаться. Это поведение описано в [документации GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html). Не перебирать seed ради удобных метрик или распределений.
5. На train выполнить `GroupKFold(n_splits=5)` с тем же model_group, без shuffle; явно сохранить fold assignment. [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) удерживает группы целиком в folds. Если независимых групп меньше пяти, остановиться и пересмотреть достаточность датасета, а не менять протокол скрыто.
6. Сохранить `vehicle_id, model_group, split, cv_fold`, где test имеет пустой cv_fold. Сохранить hash dataset, mapping, config, версии sklearn/Python и алгоритм split в metadata.

Обязательные asserts: train и test IDs не пересекаются; группы не пересекаются; все retained IDs покрыты ровно один раз; CV valid folds полностью внутри train, группы train/valid не пересекаются; все folds непустые; fingerprint совпадает. После snapshot freeze нигде не вызывать новый split вместо чтения manifest.

Проверить coverage по годам, маркам, классам и размерам групп без изменения split по test errors. Не все производители обязательно встретятся в обеих частях. Это нужно честно показать. `base_model`, `model_group`, ID не входят в predictors.

## Preprocessing

Один исходный набор X для всех моделей Midterm. Числа: model_year, displacement_l, cylinders. Категории: manufacturer, transmission, drivetrain, vehicle_class.

- Числа: `SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)`, затем `StandardScaler` для LR и KNN; можно использовать тот же scaler для дерева ради простоты общей реализации.
- Категории: missing values предварительно привести к одному null representation; `SimpleImputer(strategy='constant', fill_value='Unknown', keep_empty_features=True)`, затем `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- Объединение в `ColumnTransformer`, за ним estimator в `Pipeline`. Dense encoding разумен для небольшого structured space; измерить размер до выделения памяти.
- Во время каждого CV fold pipeline создается и fit выполняется только на fold train. На test вызывается лишь predict/transform обученного на всем train pipeline. Такой способ разделения fit и transform соответствует [рекомендациям sklearn по предотвращению утечек](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

Совместимость названий аргументов зависит от установленной версии sklearn; проверить при реализации и зафиксировать точные версии. Не добавлять статистики всего dataset, target encoding или признаки из economy measurements.

## Midterm модели

Алгоритмы проверены по предоставленным лекциям. [Lecture 3](../references/Lecture_3.pdf): Linear Regression, PDF стр.28 и 32; multiple regression стр.35–37. [Lecture 4](../references/Lecture_4.pdf): regression tree стр.15 и 40, KNN regression стр.25 и 28. Номера обозначают страницы PDF, не произвольно восстановленные slide labels.

| Модель | Начальная настройка | Зачем |
|---|---|---|
| DummyRegressor | strategy='median' | Константный ориентир для MAE, fit только по train target |
| LinearRegression | fit_intercept=True | Понятная аддитивная зависимость, реализация multiple regression |
| KNeighborsRegressor | n_neighbors=15, weights='distance', p=2 | Локальная нелинейная связь похожих характеристик |
| DecisionTreeRegressor | max_depth=8, min_samples_leaf=10, random_state=42 | Нелинейные пороги и взаимодействия без предположения линейности |

Параметры KNN и дерева — выбранная отправная точка проекта, не предписания лекций и не уже найденный optimum. Они ограничивают чувствительность к единичным наблюдениям. Для Midterm допускается малое заранее записанное CV-сравнение k=[5,15,31] и depth=[4,8,None], min_samples_leaf=[5,10,20], если достаточно времени; вся процедура выбора только внутри train. Для простого первого воспроизводимого результата использовать фиксированные начальные значения, а tuning вынести на Endterm.

Dummy оценивается на тех же folds/test, отдельным estimator на безопасном X; лишние transformations для него не обязательны. Ни Ridge вместо LinearRegression, ни SVR как четвертая модель не требуются для Midterm. Ridge зарезервирован для удобного Final text baseline, не как утверждение о материале Weeks 3–4.

## Выбор и метрики

Главная метрика `MAE = mean(abs(y_true - y_pred))` в L/100 km. Она отвечает на вопрос о средней абсолютной величине ошибки в понятных пользователю единицах. RMSE сильнее выделяет крупные ошибки; R² показывает относительную объясненную вариацию и может быть отрицательным. Не использовать accuracy для regression и не выдавать R² за процент правильно предсказанных машин. Для test MSE делить сумму квадратов ошибок на количество test rows; формула с n−2 на стр.26 Lecture 3 относится к оценке дисперсии ошибок простой регрессии, а не к predictive test MSE.

В sklearn scorer `neg_mean_absolute_error` изменить знак перед выводом. CV mean — среднее пяти положительных fold MAE, spread — population std (`ddof=0`) этих значений. Std по folds не является доверительным интервалом. Отдельно можно показать pooled OOF MAE с ясной подписью: при разных размерах folds она не обязана совпадать со средним fold MAE.

Сначала сравнить CV, выбрать победителя по минимальному mean MAE, при практически равных результатах учитывать std, простоту и время. Решение зафиксировать до test evaluation. Для rubric после этого показать test scores **всех** четырех заранее определенных моделей, не выбирать новую по test.

Одна таблица результатов содержит:

```text
stage, run_id, model, feature_set, dataset_hash, split_hash,
n_train, n_test, n_folds, parameters,
cv_mae_mean, cv_mae_std, test_mae, test_rmse, test_r2,
fit_seconds, selection_protocol
```

Отдельный `fold_metrics.csv`: run_id, model, fold, n_train, n_valid, mae, rmse, r2. Отдельные predictions: vehicle_id, split/fold, y_true, y_pred, residual, abs_error, model. Заполненные фактические таблицы всех этапов находятся в [reports/tables](../reports/tables), а общая таблица — в [all_models.csv](../reports/project_v1/all_models.csv).

Улучшение относительно dummy: `(MAE_dummy - MAE_model) / MAE_dummy × 100%`. Считать отдельно для CV и test. Отрицательное улучшение — допустимый результат, его не скрывать. Метрики показывать с одинаковым округлением, хранить полную точность.

## EDA и ошибки

Все решения preprocessing и modeling опираются на train EDA/OOF. Общие counts, missingness и dataset coverage можно описывать для всего snapshot, но не использовать test target для выбора обработки.

| График или таблица | Вопрос | Следующее решение |
|---|---|---|
| Распределение target на train | Есть ли асимметрия и редкие высокие значения | Обосновать median dummy и анализ хвостов |
| Displacement против target с цветом по class | Похожа ли зависимость на линейную | Сопоставить LR и нелинейные модели |
| Target по vehicle_class, количество строк рядом | Насколько различаются классы и их поддержка | Выделить группы для анализа ошибок |
| Missingness и coverage по годам/маркам | Где источник неполный или выборка мала | Проверить collector, описать ограничения |
| Residual против predicted и actual vs predicted | Есть ли систематическое смещение | Запланировать проверку нелинейности/отсутствующих features |

Residual фиксируем как **prediction − actual**: положительное значение означает завышение расхода. Строить MAE, median absolute error, bias и n по vehicle_class и заранее заданным displacement bins: `(0,2]`, `(2,3]`, `(3,4]`, `(4,+∞)` литров, missing отдельной группой. Показывать n; при n<30 явно отметить малую поддержку и не ранжировать надежность как установленный факт.

Сохранить 5–10 реальных строк с большой ошибкой: ID, исходные характеристики, y, prediction, error. Отделять наблюдаемое от объяснительной гипотезы, например отсутствующая масса или мощность двигателя. Feature importance и корреляция не доказывают причинность.

## Повторное использование test

Course Guide требует тот же split на Endterm и Final. После Midterm test является раскрытым сравнительным benchmark, а не полностью новым holdout. Не выполнять tuning или перебор решений по нему; новые решения обосновывать train OOF/CV и общими ограничениями. В финальном отчете признать, что многократный просмотр test может влиять на исследовательские решения.

Зафиксировать benchmark dataset до Midterm; цель 3 000 лучше выполнить до первого freeze. Если потом собраны новые строки, не дописывать их незаметно в основное обучение: это меняет размер данных и сравнение методов. Расширенный dataset — отдельный эксперимент; группы исходного test никогда не переходят в train. Для общей таблицы этапов использовать исходный frozen benchmark.

## Endterm и Final

Ensemble tuning: ограниченный search budget, `groups` передаются во все уровни CV. Для честного CV tuning-процедуры использовать nested GroupKFold; если показывается best_score_ search, подписать **selection CV**, а не unbiased CV. Test остается один и тот же.

Clustering/PCA не получают target при fit. Их не надо встраивать в регрессию только ради наличия. Для MLP следить, что встроенный random validation split может нарушить групповое разделение: использовать явный grouped validation либо отключить внутреннее early_stopping и выбрать число итераций/регуляризацию по group CV.

Final feature sets на одинаковых IDs:

| Вариант | Structured | Model name | Engine description |
|---|---|---|---|
| S | Да | Нет | Нет |
| S+M | Да | Да | Нет |
| S+E | Да | Нет | Да |
| S+M+E | Да | Да | Да |

Использовать Ridge как общее семейство для контролируемой ablation, с одинаковым бюджетом подбора по grouped train CV. Structured-only лучший ensemble из Endterm остается дополнительным ориентиром. TF-IDF fit отдельно в каждом fold; word ngrams=(1,2) и char_wb ngrams=(3,5) — кандидаты, выбирать по CV. Векторизаторы должны корректно обрабатывать пустые strings; если engine vocabulary целиком пуст в fold, применить задокументированный fallback без этого блока. Не исключать строки с отсутствующим текстом.

Публиковать paired per-row difference абсолютных ошибок. При необходимости интервала разницы MAE использовать bootstrap по model_group, а не независимым строкам; это дополнительная проверка, не требование Midterm. Сравнение текста на unseen families может показать небольшой эффект — такой вывод допустим.

Фактически Midterm использовал четыре фиксированные настройки из таблицы, без дополнительного tuning. Endterm выполнил nested grouped CV для заранее заданных RF/ExtraTrees/MLP grids; MLP обучался с `early_stopping=False`, PCA и KMeans — только на train. Outer CV оценивает search внутри каждого семейства моделей; последующий выбор семейства по тем же оценкам также вносит selection optimism. Final использовал только word TF-IDF `(1,2)`, до 5 000 признаков на текстовый блок, и одинаковый Ridge grid `alpha=[0.1,1,10,100]` для четырех вариантов; char TF-IDF и bootstrap-интервалы не выполнялись. Точные grids сохранены в [endterm.json](../configs/endterm.json) и [final.json](../configs/final.json). Выбранные по CV ensemble и text-вариант не улучшили соответствующие test baselines; этот отрицательный результат сохранен в отчете.
