"""Build an offline Markdown report and figure assets from an already completed run.

This script never retrains models or changes a saved experiment. Generated statements
use observed values; causal explanations and future improvements remain hypotheses.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from fuel_consumption.split import load_frozen_split, read_dataset
from fuel_consumption.utils import project_root, sha256_file, write_json


def markdown_table(frame: pd.DataFrame) -> str:
    """Avoid an optional tabulate dependency; escape values for ordinary GFM tables."""
    def value(item):
        if isinstance(item, float):
            return f'{item:.4f}'
        return str(item).replace('|', '\\|').replace('\n', ' ')
    header = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    separator = '| ' + ' | '.join('---' for _ in frame.columns) + ' |'
    rows = ['| ' + ' | '.join(value(item) for item in row) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join([header, separator] + rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='data/processed/vehicles.parquet')
    parser.add_argument('--split-dir', default='data/splits')
    parser.add_argument('--config', default='configs/project.json')
    parser.add_argument('--run-dir', required=True, help='Completed models/RUN_ID path')
    parser.add_argument('--audit-dir', help='Defaults to interim beside processed data')
    parser.add_argument('--output-dir', required=True, help='New report directory; refuses overwrite')
    parser.add_argument('--allow-small', action='store_true')
    args = parser.parse_args()
    root = project_root()
    dataset_path, split_dir, config_path = root / args.dataset, root / args.split_dir, root / args.config
    run_dir, output = root / args.run_dir, root / args.output_dir
    if output.exists():
        raise ValueError('Report output directory already exists; choose a new path')
    run = json.loads((run_dir / 'run_metadata.json').read_text(encoding='utf-8'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if run['status'] != 'completed' or run['test_used_for_selection'] or run['selection_source'] != 'train_group_cv':
        raise ValueError('Only a completed train-CV-selected run can be reported')
    if run['development_small_dataset'] and not args.allow_small:
        raise ValueError('Use --allow-small only for a labelled technical report')
    if sha256_file(dataset_path) != run['dataset_sha256'] or sha256_file(config_path) != run['config_sha256']:
        raise ValueError('Dataset/config differs from the saved run')
    manifest, split = load_frozen_split(dataset_path, config_path, split_dir)
    if sha256_file(split_dir / 'split_manifest.csv') != run['split_sha256']:
        raise ValueError('Split differs from the saved run')
    df = read_dataset(dataset_path)
    train = df.merge(manifest.loc[manifest['split'].eq('train'), ['vehicle_id']], on='vehicle_id', validate='one_to_one')
    tables = run_dir.resolve().parent.parent / Path(run['reports_path'].replace('\\', '/'))
    if sha256_file(tables / 'predictions.csv') != run['predictions_sha256']:
        raise ValueError('Saved predictions checksum changed')
    metrics = pd.read_csv(tables / 'metrics.csv')
    if set(metrics.model) != {'dummy', 'linear', 'knn', 'tree'} or len(metrics) != 4:
        raise ValueError('Expected one row for each of the four Midterm models')
    selected = metrics.loc[metrics.model.eq(run['selected_model'])].iloc[0]
    predictions = pd.read_csv(tables / 'predictions.csv', dtype={'vehicle_id': str})
    best = predictions.loc[predictions.model.eq(run['selected_model'])]
    subgroup = pd.read_csv(tables / 'subgroup_errors.csv')
    audit_dir = root / args.audit_dir if args.audit_dir else dataset_path.parent.parent / 'interim'
    summary_path = audit_dir / 'cleaned_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else None
    if summary and summary['cleaned_rows'] != len(df):
        raise ValueError('Cleaning audit row count differs')
    output.mkdir(parents=True)
    figures = output / 'figures'
    figures.mkdir()
    plt.rcParams.update({'figure.dpi': 140, 'axes.grid': False})
    target = config['target']['name']

    def save(figure, name):
        figure.tight_layout()
        figure.savefig(figures / name, bbox_inches='tight')
        plt.close(figure)

    figure, ax = plt.subplots(figsize=(8, 4))
    train[target].hist(bins=25, ax=ax, color='#2874a6', edgecolor='white')
    ax.axvline(train[target].median(), color='#d35400', linestyle='--', label='Train median')
    ax.set(xlabel='EPA combined consumption (L/100 km)', ylabel='Configurations', title='Training target distribution')
    ax.legend()
    save(figure, '01_train_target.png')
    figure, ax = plt.subplots(figsize=(9, 5))
    class_groups = list(train.groupby('vehicle_class', sort=True))
    class_palette = plt.get_cmap('tab20', len(class_groups))
    for index, (label, group) in enumerate(class_groups):
        ax.scatter(group.displacement_l, group[target], alpha=.5, s=15, label=label, color=class_palette(index))
    ax.set(xlabel='Engine displacement (L)', ylabel='EPA combined (L/100 km)', title='Training displacement and target')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1))
    save(figure, '02_train_displacement.png')
    classes = train.groupby('vehicle_class')[target].agg(['count', 'median', 'mean']).sort_values('count', ascending=False)
    figure, axes = plt.subplots(1, 2, figsize=(13, max(4, .35 * len(classes))))
    axes[0].boxplot([train.loc[train.vehicle_class.eq(label), target].to_numpy() for label in classes.index],
                    orientation='horizontal', tick_labels=classes.index, showfliers=True)
    axes[0].invert_yaxis()
    axes[0].set(xlabel='EPA combined (L/100 km)', title='Target by training class')
    axes[1].barh(classes.index, classes['count'].to_numpy(), color='#2874a6')
    axes[1].invert_yaxis()
    axes[1].set(xlabel='Configurations', title='Training class support')
    save(figure, '03_train_classes.png')
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, split_name in enumerate(['train_oof', 'test']):
        group = best.loc[best['split'].eq(split_name)]
        axes[row, 0].scatter(group.y_true, group.y_pred, alpha=.5, s=18)
        bounds = [min(group.y_true.min(), group.y_pred.min()), max(group.y_true.max(), group.y_pred.max())]
        axes[row, 0].plot(bounds, bounds, 'k--')
        axes[row, 0].set(xlabel='Actual (L/100 km)', ylabel='Predicted (L/100 km)', title=f'{run["selected_model"]}: {split_name}')
        axes[row, 1].scatter(group.y_pred, group.residual, alpha=.5, s=18)
        axes[row, 1].axhline(0, color='black', linestyle='--')
        axes[row, 1].set(xlabel='Predicted (L/100 km)', ylabel='Prediction - actual (L/100 km)', title=f'{split_name} residuals')
    save(figure, '04_residuals.png')
    metrics.to_csv(output / 'metrics.csv', index=False)
    subgroup.to_csv(output / 'subgroup_errors.csv', index=False)
    errors = pd.read_csv(tables / 'large_errors.csv', dtype={'vehicle_id': str})
    errors.loc[errors.model.eq(run['selected_model'])].to_csv(output / 'selected_large_errors.csv', index=False)

    n_small = int((classes['count'] < config['evaluation']['small_subgroup_threshold']).sum())
    correlation = train[['displacement_l', target]].dropna().corr().iloc[0, 1]
    stage = '**Технический прогон, меньше 1 000 строк; это не достаточный Midterm benchmark.**\n\n' if run['development_small_dataset'] else ''
    lines = [
        '# EPA fuel consumption — результаты Midterm\n', stage,
        'SE-2421: Tsybus Nikita, Bakytzhan Kassymgali.\n',
        f'Вопрос: как точно можно предсказать EPA combined расход по семи техническим спецификациям? '
        f'Для этого frozen snapshot модель **{run["selected_model"]}**, выбранная по train CV, имеет '
        f'**test MAE {selected.test_mae:.3f} L/100 km** на {split["n_test_groups"]} удержанных семействах '
        f'({split["n_test"]:,} конфигураций). Это средняя ошибка стандартизованного EPA target, '
        'не гарантия для отдельной машины и не фактический дорожный расход.\n',
        f'## Данные и split\n\nRetained rows: {len(df):,}; производителей: {df.manufacturer.nunique()}; '
        f'классов: {df.vehicle_class.nunique()}; годы {int(df.model_year.min())}–{int(df.model_year.max())}. '
        f'Train: {split["n_train"]:,} rows / {split["n_train_groups"]} families. '
        f'Test: {split["n_test"]:,} rows / {split["n_test_groups"]} families. '
        f'20% относятся к группам; фактическая test row fraction {split["test_row_fraction"]:.1%}. '
        'Grouping — manufacturer + baseModel через все годы, seed=42; train GroupKFold=5.\n',
        f'Очистка: {summary["raw_vehicle_files"]:,} raw → {summary["cleaned_rows"]:,} retained; '
        f'{summary["excluded_records"]:,} exclusions; {summary["confirmed_duplicate_records_removed"]} confirmed duplicates removed. '
        f'Unresolved candidate groups: {summary["unresolved_candidate_groups"]}. '
        'Полный audit читается в notebook; независимые причины исключения пересекаются.\n' if summary else
        '**Пробел доказательств:** cleaning summary отсутствует.\n',
        'Target = 235.2145833333333 / comb08 (US MPG). Другие economy/cost/emissions/score поля, '
        'ID, grouping и text не входят в Midterm X. Все imputers/encoders/scalers fit внутри training fold pipeline.\n',
        '## Train EDA\n\n![Target](figures/01_train_target.png)\n',
        f'Train median {train[target].median():.2f}, mean {train[target].mean():.2f} L/100 km. '
        'Median обосновывает constant MAE baseline; допустимые хвостовые значения не удаляются только за высокий расход.\n',
        '![Displacement](figures/02_train_displacement.png)\n',
        f'Pearson r={correlation:.3f} — ассоциация, не причинность или доказательство линейности. '
        'Сравнение LR/KNN/tree проверяет различные формы зависимости на одинаковых CV folds.\n',
        '![Classes](figures/03_train_classes.png)\n',
        f'Самый представленный train class: {classes.index[0]} (n={int(classes.iloc[0]["count"])}). '
        f'{n_small} классов имеют train n<30; subgroup conclusions требуют размеров групп.\n',
        '## Сравнение моделей\n\n',
        markdown_table(metrics[['model', 'cv_mae_mean', 'cv_mae_std', 'test_mae', 'test_rmse', 'test_r2',
                                'cv_improvement_over_dummy_pct', 'test_improvement_over_dummy_pct', 'selected_by_cv']]),
        f'\nВыбор **{run["selected_model"]}** сделан по mean train CV до test prediction. '
        'Test показан для всех четырех заранее заданных моделей; winner не меняется по test. '
        'CV std (ddof=0) не confidence interval. R² не процент правильных прогнозов.\n',
        '## Ошибки\n\n![Residuals](figures/04_residuals.png)\n',
        'Residual = prediction − actual. Все OOF/test subgroup estimates с n, MAE, median error и bias: '
        '[subgroup_errors.csv](subgroup_errors.csv). Selected-model реальные test ошибки: '
        '[selected_large_errors.csv](selected_large_errors.csv). При n<30 надежность групп не ранжируется как установленный факт.\n',
        '## Ограничения и следующие проверки\n\n',
        '- Конфигурации не взвешены по продажам; accuracy относится к текущему покрытию каталога.\n'
        '- Group split измеряет перенос на удержанные семейства, не будущие годы; родственные платформы могут остаться связаны.\n'
        '- comb08 округлен; EPA оценка не заменяет наблюдение поведения водителя.\n'
        '- Масса/мощность отсутствуют среди predictors; объяснение ошибок ими остается гипотезой.\n'
        '- Ensembles/tuning проверяются на train grouped CV; clustering/PCA fit без target; MLP требует grouped validation.\n'
        '- На Final сравниваются structured и model/engine text на тех же IDs, с fold-fit TF-IDF и leakage sanitation.\n'
        '- Test после Midterm раскрыт; используем те же IDs для сравнения этапов, но не для tuning.\n',
        f'## Трассировка\n\nRun `{run["run_id"]}`; dataset SHA256 `{run["dataset_sha256"]}`; '
        f'split SHA256 `{run["split_sha256"]}`. Generated report не выполняет retraining. '
        'Вклад людей/AI фиксируется в docs/CONTRIBUTIONS.md; этот отчет не подтверждает слайды или репетицию защиты.\n'
    ]
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(output / 'report_manifest.json', {'run_id': run['run_id'], 'dataset_sha256': run['dataset_sha256'],
               'split_sha256': run['split_sha256'], 'selected_model': run['selected_model'], 'retrained': False,
               'development_small_dataset': run['development_small_dataset'],
               'artifacts_sha256': {path.relative_to(output).as_posix(): sha256_file(path)
                                    for path in output.rglob('*') if path.is_file()}})
    print(output / 'report.md')


if __name__ == '__main__':
    main()
