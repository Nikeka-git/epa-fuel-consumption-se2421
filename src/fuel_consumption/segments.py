"""Training-only PCA and clustering; target summaries are computed after selection."""
from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .features import build_preprocessor, select_features
from .utils import sha256_file, write_json


def fit_segments(train: pd.DataFrame, test: pd.DataFrame, config: dict,
                 settings: dict, output_dir: str | Path) -> dict:
    """Fit on train X, select k by training silhouette, then describe both splits.

    Grouping or target columns may be present in the frames but strict feature
    selection removes them before every unsupervised fit. Distances depend on
    the numeric scaling and one-hot categorical representation, not car sales.
    """
    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"Segment output already exists: {output}")
    retained = float(settings["pca_retained_variance"])
    minimum, maximum = int(settings["min_k"]), int(settings["max_k"])
    sample_size = int(settings["silhouette_sample_size"])
    if not 0 < retained < 1 or not 2 <= minimum <= maximum <= 6 or sample_size < 3:
        raise ValueError("Segments require variance in (0,1), 2 <= min_k <= max_k <= 6 and sample >=3")
    if len(train) <= minimum or train.vehicle_id.duplicated().any() or test.vehicle_id.duplicated().any():
        raise ValueError("Segments need enough training rows and unique IDs in each split")
    if set(train.vehicle_id.astype(str)) & set(test.vehicle_id.astype(str)):
        raise ValueError("Segment train/test IDs overlap")
    X_train, X_test = select_features(train, config), select_features(test, config)
    preprocessing = build_preprocessor(config)
    transformed = preprocessing.fit_transform(X_train)
    if not np.isfinite(transformed).all() or np.var(transformed, axis=0).sum() == 0:
        raise ValueError("Segments require finite features with nonzero training variance")
    projection = PCA(n_components=retained, svd_solver="full")
    train_pc = projection.fit_transform(transformed)
    test_pc = projection.transform(preprocessing.transform(X_test))
    rng = np.random.default_rng(int(settings["random_state"]))
    sample_ids = np.sort(rng.choice(len(train), size=min(len(train), sample_size), replace=False))
    candidates, fitted, notices = [], {}, []
    for k in range(minimum, min(maximum, len(train) - 1) + 1):
        clustering = KMeans(n_clusters=k, n_init=int(settings.get("n_init", 10)),
                            random_state=int(settings["random_state"]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            labels = clustering.fit_predict(train_pc)
        notices.extend({"k": k, "category": warning.category.__name__, "message": str(warning.message)} for warning in caught)
        sampled_labels = labels[sample_ids]
        if 2 <= np.unique(sampled_labels).size < len(sample_ids):
            score = float(silhouette_score(train_pc[sample_ids], sampled_labels))
            candidates.append({"k": k, "silhouette": score, "observed_clusters": int(np.unique(labels).size),
                               "sample_n": len(sample_ids), "status": "valid"})
            fitted[k] = clustering
        else:
            candidates.append({"k": k, "silhouette": None, "observed_clusters": int(np.unique(labels).size),
                               "sample_n": len(sample_ids), "status": "invalid_silhouette_label_count"})
    valid = [row for row in candidates if row["status"] == "valid"]
    if not valid:
        raise ValueError("No valid training silhouette estimate; inspect feature variability")
    selected = min(valid, key=lambda row: (-row["silhouette"], row["k"]))
    clustering = fitted[selected["k"]]
    train_labels, test_labels = clustering.labels_, clustering.predict(test_pc)
    output.mkdir(parents=True)
    assignments = []
    for frame, split, pc, labels in [(train, "train", train_pc, train_labels), (test, "test", test_pc, test_labels)]:
        rows = pd.DataFrame({"vehicle_id": frame.vehicle_id.astype(str).to_numpy(), "split": split, "cluster": labels,
                             "pc1": pc[:, 0], "pc2": pc[:, 1] if pc.shape[1] > 1 else np.zeros(len(frame))})
        for component in range(2, pc.shape[1]):
            rows[f"pc{component + 1}"] = pc[:, component]
        assignments.append(rows)
    assignments = pd.concat(assignments, ignore_index=True)
    assignments.to_csv(output / "assignments.csv", index=False)
    # Neither target nor class descriptions enter the fit or k-selection above.
    descriptions = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    descriptions["vehicle_id"] = descriptions.vehicle_id.astype(str)
    descriptions = descriptions.merge(assignments[["vehicle_id", "cluster"]], on="vehicle_id", validate="one_to_one")
    target = config["target"]["name"]
    summary = descriptions.groupby(["split", "cluster"], observed=True).agg(
        n=("vehicle_id", "size"), displacement_mean_l=("displacement_l", "mean"),
        displacement_median_l=("displacement_l", "median"), cylinders_mean=("cylinders", "mean"),
        target_mean_l100km=(target, "mean"), target_median_l100km=(target, "median"))
    summary.to_csv(output / "cluster_summary.csv")
    descriptions.groupby(["split", "cluster", "vehicle_class"], observed=True).size().rename("n").to_csv(output / "cluster_class_counts.csv")
    pd.DataFrame(candidates).to_csv(output / "silhouette_scores.csv", index=False)
    pd.DataFrame(notices, columns=["k", "category", "message"]).to_csv(output / "warnings.csv", index=False)
    pd.DataFrame(projection.components_, columns=preprocessing.get_feature_names_out()).rename_axis("component").to_csv(output / "pca_loadings.csv")
    for name, fitted_object in [("preprocessor", preprocessing), ("pca", projection), ("kmeans", clustering)]:
        joblib.dump(fitted_object, output / f"{name}.joblib", compress=3)
    metadata = {"selected_k": selected["k"], "training_silhouette": selected["silhouette"],
                "selection_source": "training_silhouette_without_target", "target_used_for_selection": False,
                "fit_split": "train_only", "n_train": len(train), "n_test": len(test),
                "pca_components": int(projection.n_components_),
                "explained_variance_ratio": projection.explained_variance_ratio_.tolist(),
                "pc2_padded_zero": bool(projection.n_components_ == 1), "settings": settings,
                "sample_train_ids": train.iloc[sample_ids].vehicle_id.astype(str).tolist(),
                "warning_count": len(notices),
                "limitations": ["Distances depend on scaling and one-hot categories; not market share or causal segments.",
                                "Target summaries are descriptive after selecting clusters; they do not validate regression accuracy."],
                "artifacts_sha256": {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}}
    write_json(output / "segment_metadata.json", metadata)
    return metadata
