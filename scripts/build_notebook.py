"""Create an offline, top-to-bottom notebook for a collected dataset and frozen split."""
from __future__ import annotations

import argparse
from pathlib import Path
import nbformat as nbf
from fuel_consumption.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='data/processed/vehicles.parquet')
    parser.add_argument('--split-dir', default='data/splits')
    parser.add_argument('--output', default='notebooks/01_midterm.ipynb')
    parser.add_argument('--allow-small', action='store_true')
    args = parser.parse_args()
    stage_note = ('**Текущий notebook — технический прогон на ограниченном реальном сборе.** '
                  'До полного Midterm нужны минимум 1 000 валидных конфигураций, audit покрытия и data card. '
                  if args.allow_small else
                  'Для итогового Midterm должны быть подтверждены минимум 1 000 валидных конфигураций, '
                  'audit покрытия и полная data card. ')
    conclusion = ('Этот прогон проверяет работоспособность pipeline. Его метрики не используются для утверждения '
                  'точности на полном каталоге. Следующий шаг — полный collection, не менее 1 000 пригодных '
                  'конфигураций, audit категорий/дублей и новый benchmark freeze. '
                  if args.allow_small else
                  'Сформулируйте ответ на вопрос по фактическим MAE и ограничениям этого snapshot. '
                  'Привяжите Endterm эксперименты к обнаруженным на train OOF слабостям. ')
    audit_location = str(Path(args.dataset).parent.parent / 'interim')
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [
        md('# Прогноз EPA combined fuel consumption\n\n'
           'SE-2421 — Tsybus Nikita и Bakytzhan Kassymgali. '
           'Сравниваем характеристики бензиновых автомобилей и SUV рынка США. '
           'Цель — стандартизованная EPA combined оценка в L/100 km.\n\n'
           + stage_note +
           'Source: https://www.fueleconomy.gov/feg/ws/index.shtml. '
           'Notebook работает offline и заново обучает модели во временной папке.'),
        code('%matplotlib inline\nfrom pathlib import Path\nimport tempfile\nimport json\nimport pandas as pd\n'
             'import matplotlib.pyplot as plt\nfrom IPython.display import display\n'
             'from fuel_consumption.utils import project_root, load_config\n'
             'from fuel_consumption.split import read_dataset, load_frozen_split\n'
             'from fuel_consumption.train import run_midterm\n'
             'ROOT = project_root()\n'
             f'DATASET = ROOT / {args.dataset!r}\nSPLIT_DIR = ROOT / {args.split_dir!r}\n'
             f'ALLOW_SMALL = {args.allow_small!r}\n'
             'CONFIG = ROOT / "configs/project.json"\nconfig = load_config(CONFIG)\n'
             'df = read_dataset(DATASET)\nmanifest, split_info = load_frozen_split(DATASET, CONFIG, SPLIT_DIR)\n'
             'merged = df.merge(manifest[["vehicle_id", "split", "cv_fold"]], on="vehicle_id", validate="one_to_one")\n'
             'train = merged.loc[merged["split"].eq("train")].copy()\n'
             'print({"rows": len(df), "train": len(train), "test": len(df)-len(train), "development": len(df)<1000})\n'
             'display(df.groupby(["model_year", "manufacturer"]).size().rename("rows").to_frame())'),
        md('## Данные и очистка\n\n'
           'Одна строка — отдельный EPA vehicle ID после проверок source, powertrain, класса и target. '
           'Оригинальный текст сохранен для Final. Очистка и причины исключения находятся в '
           f'`{audit_location}`. Таблица ниже показывает реальные пропуски. '
           'Target не импутируется; comb08 преобразуется формулой 235.2145833333333 / MPG. '
           'MPG, стоимость, emissions, scores и идентификаторы не входят в predictors.'),
        code('feature_cols = config["features"]["numeric"] + config["features"]["categorical"]\n'
             'display(df[feature_cols + ["model_name", "engine_description"]].isna().mean().rename("missing_fraction").to_frame())\n'
             'display(pd.Series(split_info).to_frame("value"))'),
        md('## EDA на training split\n\n'
           'Исследуем форму распределения target, связь с displacement и покрытие классов. '
           'Эти графики помогают обосновать median baseline, сравнение линейной и нелинейных моделей '
           'и группы последующего анализа ошибок. Небольшой ограниченный snapshot не отражает весь рынок.'),
        code('fig, ax = plt.subplots(figsize=(8, 4))\n'
             'train["target_l100km"].hist(bins=15, ax=ax)\n'
             'ax.set(xlabel="EPA combined fuel consumption (L/100 km)", ylabel="Configurations", title="Training target distribution")\n'
             'plt.show()\n'
             'print("Median:", round(train.target_l100km.median(), 3), "Mean:", round(train.target_l100km.mean(), 3))\n'
             'print("Median dummy is the constant baseline for absolute error.")'),
        code('fig, ax = plt.subplots(figsize=(8, 4))\n'
             'for label, group in train.groupby("vehicle_class"):\n'
             '    ax.scatter(group.displacement_l, group.target_l100km, label=label, alpha=.7)\n'
             'ax.set(xlabel="Engine displacement (L)", ylabel="EPA combined (L/100 km)", title="Displacement and fuel consumption on train")\n'
             'ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))\nplt.tight_layout(); plt.show()\n'
             'print("Train Pearson correlation:", train[["displacement_l", "target_l100km"]].corr().iloc[0,1])\n'
             'print("This is association; we compare LR, KNN and tree to check model form.")'),
        code('class_summary = train.groupby("vehicle_class").target_l100km.agg(["count", "median", "mean"]).sort_values("count", ascending=False)\n'
             'display(class_summary)\n'
             'fig, ax = plt.subplots(figsize=(9, 4))\nclass_summary["count"].plot.bar(ax=ax)\n'
             'ax.set(ylabel="Training configurations", title="Vehicle-class support")\n'
             'plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.show()\n'
             'print("Classes with few rows need explicitly qualified error estimates.")'),
        md('## Фиксированные модели и train CV\n\n'
           'Dummy median, Linear Regression, KNN k=15 с distance weights и Decision Tree depth=8/min leaf=10. '
           'Числа импутируются и масштабируются, категории кодируются в sklearn pipeline внутри каждого fold. '
           'Семейства make + baseModel разделены между train/test и CV folds. '
           'Победитель выбирается по mean train CV MAE до вычисления test scores. '
           'Std по folds не является доверительным интервалом.'),
        code('(ROOT / "work").mkdir(exist_ok=True)\n'
             'temporary_run = tempfile.TemporaryDirectory(prefix="notebook_", dir=ROOT / "work")\n'
             'RUN_ROOT = Path(temporary_run.name)\n'
             'run_info = run_midterm(DATASET, CONFIG, "notebook_run", split_dir=SPLIT_DIR, output_root=RUN_ROOT, allow_small=ALLOW_SMALL)\n'
             'TABLES = RUN_ROOT / "reports" / "tables" / "notebook_run"\n'
             'metrics = pd.read_csv(TABLES / "metrics.csv")\n'
             'display(metrics[["model", "cv_mae_mean", "cv_mae_std", "test_mae", "test_rmse", "test_r2", "selected_by_cv"]])\n'
             'print("Selected by train CV:", run_info["selected_model"])'),
        md('## Анализ ошибок\n\n'
           'Остаток = prediction − actual: положительный означает завышение. '
           'Показываем размеры групп рядом с MAE и bias; при n<30 выводы о надежности слабые. '
           'Реальные трудные конфигурации дают основания для гипотез о недостающих признаках, '
           'но не доказывают причину ошибки.'),
        code('predictions = pd.read_csv(TABLES / "predictions.csv")\n'
             'best = predictions.loc[predictions.model.eq(run_info["selected_model"]) & predictions["split"].eq("test")]\n'
             'fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n'
             'axes[0].scatter(best.y_true, best.y_pred); bounds=[best.y_true.min(),best.y_true.max()]\n'
             'axes[0].plot(bounds,bounds,"k--"); axes[0].set(xlabel="Actual (L/100 km)",ylabel="Predicted (L/100 km)")\n'
             'axes[1].scatter(best.y_pred,best.residual); axes[1].axhline(0,color="black",linestyle="--")\n'
             'axes[1].set(xlabel="Predicted (L/100 km)",ylabel="Residual (L/100 km)")\n'
             'plt.tight_layout(); plt.show()\n'
             'display(pd.read_csv(TABLES / "subgroup_errors.csv"))\n'
             'display(pd.read_csv(TABLES / "large_errors.csv").head(10))'),
        md('## Выводы и следующий этап\n\n'
           + conclusion +
           'Для Endterm после полноценных результатов проверим ансамбли/tuning, PCA/clustering и MLP. '
           'На Final сравним исходные model/engine text на тех же IDs. '
           'Фактический вклад и AI помощь фиксируются в docs/CONTRIBUTIONS.md.'),
        code('print({"dataset_sha256":run_info["dataset_sha256"], "split_sha256":run_info["split_sha256"],\n'
             '       "python":run_info["python_version"], "packages":run_info["package_versions"],\n'
             '       "development_small_dataset":run_info["development_small_dataset"]})')
    ]
    nb.metadata['kernelspec'] = {'display_name':'Python 3', 'language':'python', 'name':'python3'}
    nb.metadata['language_info'] = {'name':'python'}
    output = project_root() / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output)
    print(output)


if __name__ == '__main__':
    main()
