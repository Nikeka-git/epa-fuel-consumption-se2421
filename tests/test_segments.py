"""Synthetic-only tests of unsupervised target isolation and train-only fitting."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from fuel_consumption.segments import fit_segments
from fuel_consumption.utils import project_root, sha256_file
from test_modeling import config, synthetic_frame


def settings():
    result = json.loads((project_root() / 'configs/endterm.json').read_text(encoding='utf-8'))['segments']
    result.update(max_k=3, n_init=2, silhouette_sample_size=100)
    return result


def test_segments_train_only_statistics_target_invariance_and_no_overwrite(tmp_path, config, synthetic_frame):
    train, test = synthetic_frame.iloc[:96].copy(), synthetic_frame.iloc[96:].copy()
    train['displacement_l'] = [1., 3.] + [np.nan] * 94
    test['displacement_l'] = 9999.
    test['manufacturer'] = 'New held-out manufacturer'
    first = fit_segments(train, test, config, settings(), tmp_path / 'first')
    assert first['fit_split'] == 'train_only' and first['target_used_for_selection'] is False
    assert 2 <= first['selected_k'] <= 3
    preprocessing = joblib.load(tmp_path / 'first/preprocessor.joblib')
    imputer = preprocessing.named_transformers_['numeric'].named_steps['impute']
    assert imputer.statistics_[1] == pytest.approx(2.)
    labels = pd.read_csv(tmp_path / 'first/assignments.csv', dtype={'vehicle_id': str})
    assert len(labels) == len(synthetic_frame) and labels.vehicle_id.nunique() == len(labels)
    assert labels[['pc1', 'pc2']].apply(np.isfinite).all().all()
    altered_train, altered_test = train.copy(), test.copy()
    altered_train['target_l100km'] = np.arange(len(train)) * 10000.
    altered_test['target_l100km'] = -999999.
    second = fit_segments(altered_train, altered_test, config, settings(), tmp_path / 'second')
    assert first['selected_k'] == second['selected_k']
    assert first['training_silhouette'] == pytest.approx(second['training_silhouette'])
    pd.testing.assert_frame_equal(labels, pd.read_csv(tmp_path / 'second/assignments.csv', dtype={'vehicle_id': str}))
    for filename, expected in first['artifacts_sha256'].items():
        assert expected == sha256_file(tmp_path / 'first' / filename)
    with pytest.raises(ValueError, match='already exists'):
        fit_segments(train, test, config, settings(), tmp_path / 'first')


def test_segments_do_not_fit_on_test_or_use_text_and_audit_columns(tmp_path, config, synthetic_frame, monkeypatch):
    from sklearn.compose import ColumnTransformer
    from fuel_consumption.features import MIDTERM_FEATURES
    train, test = synthetic_frame.iloc[:96].copy(), synthetic_frame.iloc[96:].copy()
    observed = []
    original = ColumnTransformer.fit_transform
    def observed_fit(self, X, *args, **kwargs):
        observed.append((len(X), tuple(X.columns)))
        assert len(X) == len(train) and tuple(X.columns) == MIDTERM_FEATURES
        return original(self, X, *args, **kwargs)
    monkeypatch.setattr(ColumnTransformer, 'fit_transform', observed_fit)
    fit_segments(train, test, config, settings(), tmp_path / 'segments')
    assert observed == [(len(train), MIDTERM_FEATURES)]


def test_segments_guard_overlap_and_k_budget(tmp_path, config, synthetic_frame):
    train, test = synthetic_frame.iloc[:96].copy(), synthetic_frame.iloc[96:].copy()
    invalid = settings()
    invalid['max_k'] = 7
    with pytest.raises(ValueError, match='max_k'):
        fit_segments(train, test, config, invalid, tmp_path / 'invalid')
    test.loc[test.index[0], 'vehicle_id'] = train.iloc[0].vehicle_id
    with pytest.raises(ValueError, match='IDs overlap'):
        fit_segments(train, test, config, settings(), tmp_path / 'overlap')
