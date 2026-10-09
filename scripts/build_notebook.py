"""Create an offline Midterm notebook with computed findings and auditable results."""
from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent
import nbformat as nbf
from fuel_consumption.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='data/processed/vehicles.parquet')
    parser.add_argument('--split-dir', default='data/splits')
    parser.add_argument('--audit-dir', help='Defaults to interim beside processed data')
    parser.add_argument('--config', default='configs/project.json')
    parser.add_argument('--run-dir', help='Read completed models/RUN_ID; otherwise refit in a temporary directory')
    parser.add_argument('--output', default='notebooks/01_midterm.ipynb')
    parser.add_argument('--allow-small', action='store_true')
    args = parser.parse_args()
    audit_location = args.audit_dir or (Path(args.dataset).parent.parent / 'interim').as_posix()
    stage_note = (
        '**Это технический прогон на ограниченном реальном сборе.** Его метрики не доказывают '
        'точность на полном каталоге и не заменяют минимум 1 000 пригодных конфигураций. '
        if args.allow_small else
        'Notebook проверяет достаточность числа строк и читает сохраненный grouped benchmark. '
        'Число строк само по себе не доказывает полноту API-каталога или отсутствие смещения выборки. '
    )
    nb = nbf.v4.new_notebook()

    def md(source: str):
        return nbf.v4.new_markdown_cell(dedent(source).strip())

    def code(source: str):
        return nbf.v4.new_code_cell(dedent(source).strip())

    nb.cells = [
        md('''
            # Прогноз EPA combined fuel consumption

            **SE-2421 — Tsybus Nikita и Bakytzhan Kassymgali.**

            Как точно можно предсказать EPA combined расход бензиновых автомобилей и SUV рынка США
            по объему двигателя, цилиндрам, трансмиссии, приводу, классу, производителю и модельному году?
            Одна строка — одна принятая конфигурация с EPA vehicle ID. Объект прогноза —
            стандартизованная EPA оценка, а не фактический расход конкретного водителя.
            Результат помогает сравнивать спецификации автомобилей в рамках покрытия нашей выборки.

            ''' + stage_note + '''

            Источник: https://www.fueleconomy.gov/feg/ws/index.shtml. Собственный collector получает
            индивидуальные записи, сохраняет raw bytes, manifest, даты и checksums. Notebook работает
            offline: читает завершенный run либо заново обучает четыре заранее заданные модели
            во временной папке. Критерии курса разобраны в `docs/REQUIREMENTS.md`.
        '''),
        code(f'''
            %matplotlib inline
            from pathlib import Path
            import tempfile
            import json
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from IPython.display import display, Markdown
            from fuel_consumption.utils import project_root, load_config, sha256_file
            from fuel_consumption.split import read_dataset, load_frozen_split
            from fuel_consumption.train import run_midterm

            ROOT = project_root()
            DATASET = ROOT / {args.dataset!r}
            SPLIT_DIR = ROOT / {args.split_dir!r}
            AUDIT_DIR = ROOT / {audit_location!r}
            CONFIG = ROOT / {args.config!r}
            EXISTING_RUN = {args.run_dir!r}
            ALLOW_SMALL = {args.allow_small!r}
            config = load_config(CONFIG)
            df = read_dataset(DATASET)
            if len(df) < config['scope']['minimum_distinct_rows'] and not ALLOW_SMALL:
                raise ValueError('Недостаточно строк для Midterm; --allow-small только для технической проверки')
            manifest, split_info = load_frozen_split(DATASET, CONFIG, SPLIT_DIR)
            merged = df.merge(manifest[['vehicle_id', 'split', 'cv_fold']], on='vehicle_id', validate='one_to_one')
            train = merged.loc[merged['split'].eq('train')].copy()
            feature_cols = config['features']['numeric'] + config['features']['categorical']
            target = config['target']['name']
            small_threshold = config['evaluation']['small_subgroup_threshold']
            pd.set_option('display.max_columns', 20)
            pd.set_option('display.max_rows', 100)
            plt.rcParams.update({{'figure.dpi': 120, 'axes.grid': False}})
            display(Markdown(f'**Покрытие:** {{len(df):,}} конфигураций, {{df.manufacturer.nunique()}} производителей, '
                             f'{{df.vehicle_class.nunique()}} классов, годы {{int(df.model_year.min())}}–{{int(df.model_year.max())}}. '
                             f'Train: {{len(train):,}}; test: {{len(df)-len(train):,}}.'))
            display(df.groupby(['model_year', 'manufacturer']).size().rename('rows').to_frame())
        '''),
        md('''
            ## Очистка: доказательства до и после

            Включаем только годы 2015–2025, допустимые gasoline labels, легковые классы и SUV.
            Проверяем secondary fuel, alternative-technology metadata, электрический мотор,
            PHEV-флаг и текстовые hybrid-сигналы. `startStop=Y` сам по себе не означает гибрид.
            Непригодный или отсутствующий target исключается; target никогда не импутируется.
            Сохраняем model-name и оригинальный engine-description text для Final.

            `target_l100km = 235.2145833333333 / comb08`, где `comb08` — US MPG.
            `comb08U` не заменяет target. MPG, economy/cost/emissions/score поля, ID, grouping и текст
            не входят в семь Midterm predictors. Следующие числа читаются из audit этого dataset.
        '''),
        code('''
            summary_path = AUDIT_DIR / 'cleaned_summary.json'
            cleaning_summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else None
            if cleaning_summary is None:
                display(Markdown('**Пробел в доказательствах:** cleaned_summary.json отсутствует; весь путь очистки не подтвержден.'))
            else:
                if cleaning_summary['cleaned_rows'] != len(df):
                    raise ValueError('Cleaning audit относится к другому числу строк')
                audit_hash = cleaning_summary.get('processed_parquet_sha256' if DATASET.suffix == '.parquet' else 'processed_csv_sha256')
                if audit_hash and audit_hash != sha256_file(DATASET):
                    raise ValueError('Cleaning audit checksum не совпадает с dataset')
                keys = ['snapshot_id', 'built_at_utc', 'raw_vehicle_files', 'excluded_records',
                        'confirmed_duplicate_records_removed', 'unresolved_candidate_groups', 'cleaned_rows',
                        'minimum_rows_met', 'desired_rows_met', 'collector_inventory_validation', 'category_allowlist_status']
                display(pd.Series({key: cleaning_summary.get(key) for key in keys}).to_frame('value'))
                display(Markdown(f"Из {cleaning_summary['raw_vehicle_files']:,} raw records оставлено {len(df):,}; "
                    f"исключено {cleaning_summary['excluded_records']:,}; подтвержденных полных source-дубликатов удалено "
                    f"{cleaning_summary['confirmed_duplicate_records_removed']:,}. "
                    'Совпадение семи predictors не считается достаточным доказательством дубля конфигурации.'))
                reasons = pd.Series(cleaning_summary.get('independent_reason_counts', {}), dtype='int64').sort_values(ascending=False)
                display(reasons.rename('records_with_reason').to_frame())
                display(Markdown('Причины **пересекаются**: запись может иметь несколько причин; '
                                 'их сумма не равна числу исключенных строк.'))
            for filename in ['cleaning_summary.csv', 'duplicate_resolution.csv', 'duplicate_candidates.csv']:
                path = AUDIT_DIR / filename
                if path.exists():
                    display(Markdown(f'**{filename}**'))
                    table = pd.read_csv(path)
                    display(table.head(20))
                    if len(table) > 20:
                        print(f'{len(table)} rows total; full audit: {path}')
                else:
                    print(f'Audit file absent: {filename}')
        '''),
        code('''
            def missing_counts(frame, columns):
                rows = []
                for column in columns:
                    values = frame[column]
                    null = values.isna()
                    blank = values.astype('string').str.strip().eq('').fillna(False)
                    missing = null | blank
                    rows.append({'field': column, 'null_n': int(null.sum()), 'empty_text_n': int(blank.sum()),
                                 'missing_n': int(missing.sum()), 'missing_fraction': float(missing.mean())})
                return pd.DataFrame(rows).set_index('field')

            display(Markdown('**После очистки — retained rows; описательная проверка пропусков.**'))
            retained_missing = missing_counts(df, feature_cols + ['model_name', 'engine_description', target])
            display(retained_missing)
            dictionary_path = AUDIT_DIR / 'raw_field_dictionary.json'
            if dictionary_path.exists():
                dictionary = json.loads(dictionary_path.read_text(encoding='utf-8'))
                mapped = []
                for source_field, entry in dictionary.items():
                    if entry.get('internal_field') in feature_cols + ['model_name', 'engine_description', 'combined_mpg']:
                        mapped.append({'api_field': source_field, 'internal_field': entry['internal_field'],
                                       'raw_present_n': entry['present_count'], 'raw_missing_or_empty_n': entry['missing_or_empty_count'],
                                       'disposition': entry['disposition']})
                display(Markdown('**До очистки — raw source dictionary.** Включает записи вне scope; '
                                 'знаменатель отличается от retained table.'))
                display(pd.DataFrame(mapped))
            display(Markdown('Numeric missing values импутируются median, categorical missing — Unknown, '
                             'внутри training pipeline каждого fold. Пустой текст сохраняется для Final.'))
            display(Markdown('**Feature allowlist:** ' + ', '.join(f'`{name}`' for name in feature_cols)))
        '''),
        md('''
            ## Что оценивает split

            GroupShuffleSplit seed=42 удерживает 20% **семейств**, а не ровно 20% строк.
            Группа — нормализованные manufacturer + baseModel, объединенные через все годы;
            полный model-name служит fallback и требует audit. Проверяем перенос на удержанные
            семейства в рамках собранного каталога. Это не временной прогноз нового модельного года
            и не проверка на другом источнике. Пять GroupKFold folds существуют только внутри train.

            Test не используется для выбора seed, признаков, preprocessing или гиперпараметров.
            На Endterm/Final используем те же IDs; после Midterm test уже раскрытый сравнительный benchmark.
        '''),
        code('''
            split_keys = ['frozen_at_utc', 'method', 'random_state', 'test_group_fraction', 'n_train', 'n_test',
                          'n_train_groups', 'n_test_groups', 'test_row_fraction', 'cv_method', 'cv_n_splits',
                          'train_test_id_overlap', 'train_test_group_overlap', 'cv_group_overlap',
                          'unresolved_fallback_rows', 'development_small_dataset']
            display(pd.Series({key: split_info.get(key) for key in split_keys}).to_frame('value'))
            display(pd.DataFrame.from_dict(split_info['fold_sizes'], orient='index').rename_axis('fold'))
            alias_path = SPLIT_DIR / 'group_alias_audit.csv'
            if alias_path.exists():
                alias_audit = pd.read_csv(alias_path)
                print(f'Fallback/alias audit: {len(alias_audit)} rows; unresolved: {split_info["unresolved_fallback_rows"]}')
                display(alias_audit.head(15))
            display(Markdown(f"Test содержит {split_info['n_test_groups']} семейств и "
                             f"{split_info['test_row_fraction']:.1%} retained rows. "
                             'Отсутствие буквального group overlap не доказывает отсутствия родственных платформ.'))
        '''),
        md('''
            ## EDA только на train: распределение target

            Все следующие target-графики используют training rows. Counts/missingness всего snapshot
            выше описательные; test target не используется для выбора обработки. MAE в L/100 km
            измеряет средний размер ошибки в понятных покупателю единицах. Median dummy — оптимальная
            постоянная модель для absolute error; он обучается заново в каждом fold.
        '''),
        code('''
            fig, ax = plt.subplots(figsize=(8, 4))
            train[target].hist(bins=25, ax=ax, color='#2874a6', edgecolor='white')
            ax.axvline(train[target].median(), color='#d35400', linestyle='--', label='Train median')
            ax.set(xlabel='EPA combined fuel consumption (L/100 km)', ylabel='Configurations',
                   title=f'Training target distribution (n={len(train):,})')
            ax.legend(); plt.tight_layout(); plt.show()
            quantiles = train[target].quantile([.05, .25, .5, .75, .95])
            display(quantiles.rename('L/100 km').to_frame())
            display(Markdown(f"Train median = **{train[target].median():.2f}**, mean = **{train[target].mean():.2f}** L/100 km; "
                             f"центральные 90%: {quantiles.loc[.05]:.2f}–{quantiles.loc[.95]:.2f}. "
                             'Диапазон задает масштаб MAE. Хвост не удаляется только из-за высоких значений: '
                             'они могут быть допустимыми конфигурациями; validity проверяет cleaner.'))
        '''),
        md('''
            ## EDA: объем двигателя и расход

            Scatter показывает форму связи и различия классов. Корреляция означает ассоциацию,
            а не причинный эффект изменения двигателя. Масса и мощность могут быть объяснительной
            гипотезой, но в нашем feature set они не измеряются.
        '''),
        code('''
            fig, ax = plt.subplots(figsize=(9, 5))
            class_groups = list(train.groupby('vehicle_class', sort=True))
            class_palette = plt.get_cmap('tab20', len(class_groups))
            for index, (label, group) in enumerate(class_groups):
                ax.scatter(group.displacement_l, group[target], label=label, color=class_palette(index), alpha=.5, s=15)
            ax.set(xlabel='Engine displacement (L)', ylabel='EPA combined (L/100 km)',
                   title='Displacement and consumption on training families')
            ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1))
            plt.tight_layout(); plt.show()
            association = train[['displacement_l', target]].dropna()
            correlation = association.corr().iloc[0, 1]
            display(Markdown(f"Pearson r = **{correlation:.3f}** на {len(association):,} train rows с обоими значениями. "
                             'По r нельзя подтвердить линейность или причинность. '
                             'Сравниваем Linear Regression с KNN и Decision Tree на одинаковых grouped CV folds.'))
        '''),
        md('''
            ## EDA: расход и поддержка классов

            Совмещаем распределения target и число конфигураций: группа с широкой вариацией или
            малым n может иметь нестабильную оценку ошибки. Test ошибки анализируем после выбора модели.
        '''),
        code('''
            class_summary = train.groupby('vehicle_class')[target].agg(['count', 'median', 'mean', 'std']).sort_values('count', ascending=False)
            display(class_summary)
            order = class_summary.index.tolist()
            fig, axes = plt.subplots(1, 2, figsize=(13, max(4, .35 * len(order))))
            axes[0].boxplot([train.loc[train.vehicle_class.eq(label), target].to_numpy() for label in order],
                            orientation='horizontal', tick_labels=order, showfliers=True)
            axes[0].set(xlabel='EPA combined (L/100 km)', title='Target by class on train')
            axes[1].barh(order, class_summary['count'].to_numpy(), color='#2874a6')
            axes[1].invert_yaxis(); axes[0].invert_yaxis()
            axes[1].set(xlabel='Training configurations', title='Class support')
            plt.tight_layout(); plt.show()
            largest = class_summary.iloc[0]
            few_classes = class_summary.loc[class_summary['count'].lt(small_threshold)].index.tolist()
            display(Markdown(f"Самый представленный train class — **{class_summary.index[0]}** "
                             f"({int(largest['count']):,} строк, median {largest['median']:.2f} L/100 km). "
                             f"У {len(few_classes)} классов n<{small_threshold}. "
                             'Различия классов поддерживают vehicle_class predictor и subgroup MAE analysis; '
                             'они не доказывают равного качества для всех классов.'))
        '''),
        md('''
            ## Модели и train CV

            Dummy median — ориентир без спецификаций. Linear Regression — объяснимый аддитивный
            ориентир из Week 3. KNN проверяет сходство конфигураций: k=15 снижает чувствительность
            к отдельному соседу, distance weights учитывают близость. Decision Tree моделирует
            нелинейные пороги/взаимодействия: depth=8, min leaf=10 ограничивают сложность.
            KNN и регрессионное дерево подтверждены Lecture 4.

            Imputers, numeric scaler и one-hot encoder находятся в sklearn pipeline и fit только
            на training portion fold; unknown categories допустимы. Четыре модели получают одинаковые
            семь признаков. Параметры заданы до оценки; tuning — следующий этап.

            Победитель — минимальный mean train CV MAE. `cv_selection.json` сохраняется **до** test
            prediction; test metrics показываются для всех моделей и не меняют победителя.
            Mean/std считаются по пяти fold MAE (`ddof=0`). Fold std не confidence interval;
            pooled OOF MAE может отличаться от mean fold MAE при разных fold sizes.
        '''),
        code('''
            display(pd.DataFrame([{'model': name, 'class': item['class'], 'parameters': json.dumps(item['params'], sort_keys=True)}
                                  for name, item in config['midterm_models'].items()]))
            if EXISTING_RUN is None:
                (ROOT / 'work').mkdir(exist_ok=True)
                temporary_run = tempfile.TemporaryDirectory(prefix='notebook_', dir=ROOT / 'work')
                RUN_ROOT = Path(temporary_run.name)
                run_info = run_midterm(DATASET, CONFIG, 'notebook_run', split_dir=SPLIT_DIR,
                                       output_root=RUN_ROOT, allow_small=ALLOW_SMALL)
                RUN_DIR = RUN_ROOT / 'models' / 'notebook_run'
            else:
                RUN_DIR = ROOT / EXISTING_RUN
                RUN_ROOT = RUN_DIR.resolve().parent.parent
                run_info = json.loads((RUN_DIR / 'run_metadata.json').read_text(encoding='utf-8'))
                if run_info['status'] != 'completed':
                    raise ValueError('Existing run is not completed')
                if run_info['dataset_sha256'] != sha256_file(DATASET):
                    raise ValueError('Existing run относится к другому dataset')
                if run_info['split_sha256'] != sha256_file(SPLIT_DIR / 'split_manifest.csv'):
                    raise ValueError('Existing run относится к другому split')
                if run_info['config_sha256'] != sha256_file(CONFIG):
                    raise ValueError('Existing run относится к другому config')
                if run_info['development_small_dataset'] and not ALLOW_SMALL:
                    raise ValueError('Small development run нельзя выдать за достаточный benchmark')
                print('Сохраненные результаты; повторное обучение не выполняется.')
            selection = json.loads((RUN_DIR / 'cv_selection.json').read_text(encoding='utf-8'))
            assert selection['selection_source'] == 'train_group_cv'
            assert run_info['test_used_for_selection'] is False
            TABLES = RUN_ROOT / Path(run_info['reports_path'].replace('\\\\', '/'))
            if sha256_file(TABLES / 'predictions.csv') != run_info['predictions_sha256']:
                raise ValueError('Saved predictions checksum changed')
            metrics = pd.read_csv(TABLES / 'metrics.csv')
            assert set(metrics.model) == {'dummy', 'linear', 'knn', 'tree'} and len(metrics) == 4
            display(metrics[['model', 'n_train', 'n_test', 'n_folds', 'cv_mae_mean', 'cv_mae_std',
                             'test_mae', 'test_rmse', 'test_r2', 'cv_improvement_over_dummy_pct',
                             'test_improvement_over_dummy_pct', 'selected_by_cv']].round(4))
            display(pd.read_csv(TABLES / 'fold_metrics.csv'))
            selected_model = run_info['selected_model']
            selected_metrics = metrics.loc[metrics.model.eq(selected_model)].iloc[0]
            display(Markdown(f"**По train CV выбрана {selected_model}:** MAE "
                             f"{selected_metrics.cv_mae_mean:.3f} ± {selected_metrics.cv_mae_std:.3f} L/100 km. "
                             f"CV improvement относительно dummy: {selected_metrics.cv_improvement_over_dummy_pct:.1f}%. "
                             f"Test MAE = {selected_metrics.test_mae:.3f} L/100 km, "
                             f"RMSE = {selected_metrics.test_rmse:.3f}, R² = {selected_metrics.test_r2:.3f}. "
                             'R² не процент правильно предсказанных автомобилей.'))
        '''),
        md('''
            ## Ошибки: train OOF и test

            Residual = prediction − actual: положительный означает завышение расхода.
            Train OOF прогноз каждой строки получен моделью, не обученной на ее семействе.
            OOF обосновывает следующие эксперименты; test диагностирует выбранную модель,
            но не служит подбору новых параметров.
        '''),
        code('''
            predictions = pd.read_csv(TABLES / 'predictions.csv', dtype={'vehicle_id': str})
            best = predictions.loc[predictions.model.eq(selected_model)].copy()
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            for row, split_name in enumerate(['train_oof', 'test']):
                group = best.loc[best['split'].eq(split_name)]
                axes[row, 0].scatter(group.y_true, group.y_pred, alpha=.5, s=18)
                bounds = [min(group.y_true.min(), group.y_pred.min()), max(group.y_true.max(), group.y_pred.max())]
                axes[row, 0].plot(bounds, bounds, 'k--')
                axes[row, 0].set(xlabel='Actual (L/100 km)', ylabel='Predicted (L/100 km)',
                                 title=f'{selected_model}: {split_name} (n={len(group):,})')
                axes[row, 1].scatter(group.y_pred, group.residual, alpha=.5, s=18)
                axes[row, 1].axhline(0, color='black', linestyle='--')
                axes[row, 1].set(xlabel='Predicted (L/100 km)', ylabel='Prediction - actual (L/100 km)',
                                 title=f'{split_name} residuals')
            plt.tight_layout(); plt.show()
            error_summary = best.groupby('split').agg(n=('vehicle_id', 'size'), pooled_mae=('abs_error', 'mean'),
                                                      median_absolute_error=('abs_error', 'median'), bias=('residual', 'mean'))
            display(error_summary)
            display(Markdown(f"Mean residual: train OOF {error_summary.loc['train_oof', 'bias']:.3f}, "
                             f"test {error_summary.loc['test', 'bias']:.3f} L/100 km. "
                             'Bias около нуля может скрывать противоположные ошибки разных групп.'))
        '''),
        md('''
            ## Ошибки по vehicle class и engine size

            Заранее определенные интервалы: (0,2], (2,3], (3,4], (4,+∞) L; missing отдельно.
            Для всех моделей и обоих splits показываем n, MAE, median absolute error и bias.
            При n<30 поддержка мала; описательное сравнение не доказывает надежность группы.
        '''),
        code('''
            subgroup = pd.read_csv(TABLES / 'subgroup_errors.csv')
            for field in ['vehicle_class', 'engine_size_bin']:
                display(Markdown(f'**{field}: all models, OOF + test**'))
                display(subgroup.loc[subgroup.subgroup_field.eq(field),
                                     ['model', 'split', 'subgroup', 'n', 'mae', 'median_absolute_error', 'bias', 'small_support']]
                        .sort_values(['split', 'model', 'subgroup']).reset_index(drop=True))
            selected_groups = subgroup.loc[subgroup.model.eq(selected_model)]
            for split_name in ['train_oof', 'test']:
                supported = selected_groups.loc[selected_groups['split'].eq(split_name)
                                                 & selected_groups.n.ge(small_threshold)].sort_values('mae', ascending=False)
                if supported.empty:
                    display(Markdown(f'**{split_name}:** нет estimates с n≥{small_threshold}; '
                                     'не ранжируем надежность групп как установленный факт.'))
                else:
                    worst = supported.iloc[0]
                    display(Markdown(f"**{split_name}, поддержанные группы:** наибольший наблюдаемый MAE "
                                     f"у {worst.subgroup_field} = {worst.subgroup}: {worst.mae:.3f} L/100 km "
                                     f"(n={int(worst.n)}). Это оценка текущего split, не причинный вывод."))
            large_errors = pd.read_csv(TABLES / 'large_errors.csv', dtype={'vehicle_id': str})
            display(Markdown('**Реальные test-конфигурации с наибольшей ошибкой выбранной по CV модели.**'))
            display(large_errors.loc[large_errors.model.eq(selected_model)].head(10))
            oof_examples = best.loc[best['split'].eq('train_oof')].nlargest(10, 'abs_error')
            oof_examples = oof_examples.merge(df[['vehicle_id'] + feature_cols + ['model_name']], on='vehicle_id', validate='one_to_one')
            display(Markdown('**Train OOF примеры для планирования следующих экспериментов.**'))
            display(oof_examples)
        '''),
        md('''
            ## Ответ, ограничения и следующий этап

            Сводка вычисляется из результатов; она не подменяет интерпретацию графиков командой.
            Smoke scores не называются оценкой полного рынка. Вклад участников должен быть подтвержден contribution log.
        '''),
        code(r'''
            sample_label = 'ограниченного технического snapshot' if run_info['development_small_dataset'] else 'этого frozen benchmark'
            display(Markdown(f"Для **{sample_label}** семь спецификаций и модель {selected_model}, выбранная по train CV, "
                             f"дают test MAE **{selected_metrics.test_mae:.3f} L/100 km** на "
                             f"{split_info['n_test_groups']} удержанных семействах ({split_info['n_test']:,} конфигураций). "
                             f"Изменение test MAE относительно dummy в сторону улучшения: "
                             f"{selected_metrics.test_improvement_over_dummy_pct:.1f}% (отрицательное означает ухудшение). "
                             'Это средняя ошибка EPA target, не гарантированный интервал каждой машины '
                             'и не точность реального дорожного расхода.'))
            limitations = [
                f'Покрытие: {df.manufacturer.nunique()} производителей, {df.vehicle_class.nunique()} классов, '
                f'годы {int(df.model_year.min())}–{int(df.model_year.max())}; конфигурации не взвешены по продажам.',
                'Target из rounded comb08; преобразование L/100 km нелинейно. Результаты ограничены EPA методикой/округлением.',
                'Группы из source baseModel и audit aliases; общие платформы между разными именами могут оставаться связанными.',
                'Масса, мощность, аэродинамика и поведение водителя не predictors; объяснения ошибок через них — гипотезы.',
                f'Группы с n<{small_threshold} имеют малую поддержку. Fold std не confidence interval.',
                'После Midterm test раскрыт; tuning остается на train grouped CV, сравнение этапов — на тех же test IDs.'
            ]
            if run_info['development_small_dataset']:
                limitations.insert(0, 'Меньше 1 000 retained rows: технический прогон не выполняет minimum benchmark requirement.')
            if cleaning_summary:
                limitations.append(f"Category status: {cleaning_summary.get('category_allowlist_status')}; "
                                   f"unresolved duplicate candidates: {cleaning_summary.get('unresolved_candidate_groups')}. "
                                   'Минимум строк не подтверждает complete catalogue traversal; проверяем collection manifest.')
            display(Markdown('\n'.join('- ' + text for text in limitations)))
            display(pd.DataFrame([
                {'basis / hypothesis': f'OOF MAE {error_summary.loc["train_oof", "pooled_mae"]:.3f}; possible nonlinear interactions',
                 'next experiment': 'Tuned ensembles with nested train GroupKFold',
                 'evidence': 'OOF/CV vs best Midterm; fixed test for final comparison'},
                {'basis / hypothesis': f'{len(train):,} train rows; class composition differs',
                 'next experiment': 'Clustering and PCA, fit without target',
                 'evidence': 'Stable segments/structure; no automatic MAE improvement claim'},
                {'basis / hypothesis': 'Nonlinear capacity: untested next-stage hypothesis',
                 'next experiment': 'MLP with explicit grouped validation',
                 'evidence': 'Train CV and fixed test; prevent group leakage in early stopping'},
                {'basis / hypothesis': f'Engine text missing fraction {retained_missing.loc["engine_description", "missing_fraction"]:.1%}',
                 'next experiment': 'Final structured vs model/engine text, fold-fit TF-IDF',
                 'evidence': 'Same IDs; sanitize leakage text; paired absolute-error differences'}
            ]))
            display(Markdown('Вклад и AI: `docs/CONTRIBUTIONS.md`; source/scope: `docs/DATA_CARD.md`; '
                             'course requirements и неизвестные dates/approval: `docs/REQUIREMENTS.md`. '
                             'Слайды и участие обоих студентов на защите — отдельные артефакты.'))
        '''),
        md('''
            ## Воспроизводимость

            Collection не запускается внутри notebook. Dataset/split проходят checksum validation;
            обучение сохраняет параметры, версии, CV selection до test, per-row predictions и model checksums.
            Temporary refit не перезаписывает основной run.
        '''),
        code('''
            print({'dataset_sha256': run_info['dataset_sha256'], 'split_sha256': run_info['split_sha256'],
                   'config_sha256': run_info['config_sha256'], 'selected_by': run_info['selection_source'],
                   'test_used_for_selection': run_info['test_used_for_selection'],
                   'python': run_info['python_version'], 'packages': run_info['package_versions'],
                   'development_small_dataset': run_info['development_small_dataset']})
            print('Stored tables:', TABLES)
        '''),
    ]
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.metadata['language_info'] = {'name': 'python'}
    output = project_root() / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output)
    print(output)


if __name__ == '__main__':
    main()
