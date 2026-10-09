"""Synthetic-only checks of Endterm invariants, not reportable project results."""
import copy
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from fuel_consumption.endterm import build_endterm_pipeline, candidate_parameters, run_endterm
from fuel_consumption.features import MIDTERM_FEATURES, select_features
from fuel_consumption.split import freeze_split, read_dataset
from fuel_consumption.utils import project_root, sha256_file
from test_modeling import config, synthetic_frame, _write_inputs


@pytest.fixture
def stage_config():
    stage = json.loads((project_root() / 'configs/endterm.json').read_text(encoding='utf-8'))
    stage['evaluation']['inner_n_splits'] = 2
    for model in ['random_forest', 'extra_trees']:
        stage['models'][model]['params']['n_estimators'] = 5
        stage['models'][model]['candidates'] = [{'max_depth': 3, 'min_samples_leaf': 2}]
    stage['models']['mlp']['params'].update(max_iter=3, batch_size=16)
    stage['models']['mlp']['candidates'] = [{'hidden_layer_sizes': [4], 'alpha': .01}]
    stage['segments'].update(max_k=3, n_init=2, silhouette_sample_size=120)
    return stage


def test_mlp_pipeline_fits_input_and_target_statistics_only_on_training(config, stage_config, synthetic_frame):
    training = synthetic_frame.iloc[:12].copy()
    training['displacement_l'] = [1., 3.] + [np.nan] * 10
    parameters = candidate_parameters(stage_config, 'mlp')[0]
    pipeline = build_endterm_pipeline(config, stage_config, 'mlp', parameters)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pipeline.fit(training, training.target_l100km)
    imputer = pipeline.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute']
    target_scaler = pipeline.named_steps['model'].transformer_
    assert imputer.statistics_[1] == pytest.approx(2.)
    assert target_scaler.mean_[0] == pytest.approx(training.target_l100km.mean())
    assert pipeline.named_steps['model'].regressor_.early_stopping is False
    assert tuple(pipeline.named_steps['select'].feature_names_in_) == MIDTERM_FEATURES
    held_out = synthetic_frame.iloc[12:15].copy()
    held_out['displacement_l'] = 99999.
    held_out['manufacturer'] = 'Unseen Maker'
    prediction = pipeline.predict(held_out)
    held_out['target_l100km'] = 1e9
    held_out['combined_mpg'] = -1e6
    np.testing.assert_array_equal(prediction, pipeline.predict(held_out))
    assert imputer.statistics_[1] == pytest.approx(2.)
    assert target_scaler.mean_[0] == pytest.approx(training.target_l100km.mean())


@pytest.mark.parametrize('change', ['row_validation', 'unbounded_candidates', 'unseeded', 'unscaled_target'])
def test_endterm_search_guardrails_reject_leaky_or_unbounded_settings(stage_config, change):
    altered = copy.deepcopy(stage_config)
    if change == 'row_validation':
        altered['models']['mlp']['params']['early_stopping'] = True
    elif change == 'unbounded_candidates':
        altered['models']['mlp']['candidates'] = [{'alpha': index} for index in range(13)]
    elif change == 'unseeded':
        altered['models']['mlp']['params']['random_state'] = None
    else:
        altered['models']['mlp']['target_transform'] = 'all_rows_scaler'
    with pytest.raises(ValueError):
        candidate_parameters(altered, 'mlp')


def test_endterm_runner_nested_groups_auditable_selection_hashes_and_no_overwrite(tmp_path, config, stage_config, synthetic_frame, monkeypatch):
    dataset, settings = _write_inputs(tmp_path, synthetic_frame, config)
    stage_path = tmp_path / 'endterm.json'
    stage_path.write_text(json.dumps(stage_config), encoding='utf-8')
    splits = tmp_path / 'splits'
    manifest, _ = freeze_split(dataset, settings, splits, allow_small=True)
    with pytest.raises(ValueError, match='At least 1000'):
        run_endterm(dataset, settings, stage_path, 'too_small', splits, tmp_path / 'artifacts')
    assert not (tmp_path / 'artifacts').exists()
    X_test = select_features(read_dataset(dataset).merge(manifest[['vehicle_id', 'split']], on='vehicle_id')
                             .loc[lambda frame: frame['split'].eq('test')].reset_index(drop=True), config)
    from sklearn.pipeline import Pipeline
    original_predict = Pipeline.predict
    checked_test_calls = []
    run_dir = tmp_path / 'artifacts/models/synthetic_endterm_only'
    def observed_predict(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame) and X.equals(X_test):
            selection = json.loads((run_dir / 'cv_selection.json').read_text(encoding='utf-8'))
            assert selection['test_evaluation_started'] is False
            assert selection['test_used_for_selection'] is False
            checked_test_calls.append(type(self.named_steps['model']).__name__)
        return original_predict(self, X, *args, **kwargs)
    monkeypatch.setattr(Pipeline, 'predict', observed_predict)
    result = run_endterm(dataset, settings, stage_path, 'synthetic_endterm_only', splits, tmp_path / 'artifacts', allow_small=True)
    assert len(checked_test_calls) == 3
    assert result['selection_source'] == 'train_nested_group_cv' and result['test_used_for_selection'] is False
    assert result['development_small_dataset'] and result['convergence_warning_count'] > 0
    reports = tmp_path / 'artifacts' / result['reports_path']
    metrics = pd.read_csv(reports / 'metrics.csv')
    assert set(metrics.model) == {'random_forest', 'extra_trees', 'mlp'}
    assert metrics.cv_role.eq('outer_group_cv_of_search_procedure').all()
    assert metrics.sort_values(['cv_mae_mean', 'cv_mae_std']).iloc[0].model == result['selected_model']
    search = pd.read_csv(reports / 'search_fold_metrics.csv')
    assert search.group_overlap.eq(0).all()
    assert set(search.context) == {'outer_0', 'outer_1', 'outer_2', 'outer_3', 'outer_4', 'full_train_selection'}
    assert len(search) == 3 * (5 * 2 + 5)
    folds = pd.read_csv(reports / 'fold_metrics.csv')
    assert len(folds) == 15 and folds.group_overlap.eq(0).all()
    predictions = pd.read_csv(reports / 'predictions.csv', dtype={'vehicle_id': str})
    assert len(predictions) == len(synthetic_frame) * 3
    assert predictions.groupby(['model', 'vehicle_id']).size().eq(1).all()
    assert result['predictions_sha256'] == sha256_file(reports / 'predictions.csv')
    assert result['metrics_sha256'] == sha256_file(reports / 'metrics.csv')
    for model, expected in result['model_artifact_sha256'].items():
        path = run_dir / f'{model}.joblib'
        assert expected == sha256_file(path)
        fitted = joblib.load(path)
        saved_test = predictions.loc[predictions.model.eq(model) & predictions['split'].eq('test')].sort_values('vehicle_id')
        test_rows = synthetic_frame.loc[synthetic_frame.vehicle_id.isin(saved_test.vehicle_id)].sort_values('vehicle_id')
        np.testing.assert_allclose(fitted.predict(test_rows), saved_test.y_pred)
    with pytest.raises(ValueError, match='already exists'):
        run_endterm(dataset, settings, stage_path, 'synthetic_endterm_only', splits, tmp_path / 'artifacts', allow_small=True)
    assert result['segments']['target_used_for_selection'] is False


def test_endterm_refuses_changed_frozen_dataset_before_training(tmp_path, config, stage_config, synthetic_frame):
    dataset, settings = _write_inputs(tmp_path, synthetic_frame, config)
    splits = tmp_path / 'splits'
    freeze_split(dataset, settings, splits, allow_small=True)
    stage_path = tmp_path / 'endterm.json'
    stage_path.write_text(json.dumps(stage_config), encoding='utf-8')
    dataset.write_bytes(dataset.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        run_endterm(dataset, settings, stage_path, 'tampered', splits, tmp_path / 'artifacts', allow_small=True)
    assert not (tmp_path / 'artifacts').exists()


def test_midterm_comparison_requires_same_dataset_split(tmp_path):
    from fuel_consumption.endterm import _midterm_comparison
    run = tmp_path / 'models' / 'midterm'
    run.mkdir(parents=True)
    (run / 'run_metadata.json').write_text(json.dumps({'stage': 'midterm', 'status': 'completed',
        'test_used_for_selection': False, 'dataset_sha256': 'different', 'split_sha256': 'split'}), encoding='utf-8')
    with pytest.raises(ValueError, match='same dataset'):
        _midterm_comparison(run, 'expected', 'split')
