"""
Whole-brain multiple-regression RSA using the cached neural searchlight RDMs.

Two three-predictor models are run:

    neural RDM ~ CLIP cosine RDM + GIST L2-Euclidean RDM + raw VA Euclidean RDM
    neural RDM ~ CLIP cosine RDM + GIST L2-Euclidean RDM + z-scored VA Euclidean RDM

All RDM vectors are rank transformed and standardized before regression. Therefore,
each coefficient is a standardized multiple-regression beta in a Spearman-RSA model:
it measures the predictor's unique association with the neural RDM while controlling
for the other two model RDMs.

Group inference is performed on the 20 subject-level beta maps. The primary maps use
one-sided voxelwise max-T family-wise-error correction from 10,000 synchronized
subject sign-flip permutations. Synchronized means the same sign pattern is applied
to every voxel in a permutation, preserving the observed spatial dependence between
overlapping searchlights. Secondary BH-FDR q<0.01 and q<0.001 maps with a 30-voxel
extent filter are also saved for comparison with the project's earlier maps.

The script reuses the already computed single-model Spearman score tables for CLIP,
GIST, and z-scored VA. It computes the raw-VA Spearman table once if it is absent.
"""
from __future__ import annotations

import json
import os
import pickle

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import rankdata, ttest_1samp
from sklearn.metrics import pairwise_distances
from statsmodels.stats.multitest import fdrcorrection

import common as c


PREDICTOR_NAMES = ("clip", "gist", "va")
N_PERMUTATIONS = 10_000
PERMUTATION_SEED = 20260813
FDR_ALPHAS = {"fdr01": 0.01, "fdr001": 0.001}
CLUSTER_THRESHOLD = 30

RDM_PATHS = {
    "clip": os.path.join(c.CODES_DIR, "rdm_clip_cosine.csv"),
    "gist": os.path.join(c.CODES_DIR, "rdm_gist_l2norm_euclidean.csv"),
    "va_zscore": os.path.join(c.CODES_DIR, "rdm_va_zscore.csv"),
}

EVAL_PATHS = {
    "clip": os.path.join(c.CODES_DIR, "eval_score_clip_cosine.csv"),
    "gist": os.path.join(c.CODES_DIR, "eval_score_gist_l2norm_euclidean.csv"),
    "va_zscore": os.path.join(c.CODES_DIR, "eval_score_va_zscore.csv"),
    "va_raw": os.path.join(c.CODES_DIR, "eval_score_va_raw.csv"),
}

METADATA_PATH = os.path.join(c.CODES_DIR, "SL_RDM_metadata.npz")
SUMMARY_PATH = os.path.join(c.CODES_DIR, "multiple_regression_summary.json")


def load_rdm_csv(path: str) -> np.ndarray:
    """Load a pandas-written 60x60 RDM CSV and validate its geometry."""
    rdm = pd.read_csv(path, index_col=0).to_numpy(dtype=np.float64)
    assert rdm.shape == (60, 60), f"expected a 60x60 RDM at {path}, got {rdm.shape}"
    assert np.all(np.isfinite(rdm)), f"non-finite entries in {path}"
    assert np.allclose(rdm, rdm.T, atol=1e-8), f"RDM is not symmetric: {path}"
    assert np.allclose(np.diag(rdm), 0, atol=1e-8), f"RDM diagonal is not zero: {path}"
    return rdm


def build_raw_va_rdm() -> np.ndarray:
    """Raw Euclidean distance on [arousal, valence] in canonical stimulus order."""
    stim_order = c.load_stim_order()
    conditions = pd.read_csv(c.BEHAVIORAL_CONDITIONS_PATH)
    unique = (
        conditions[["iaps_id", "Mean_Valence", "Mean_Arousal"]]
        .drop_duplicates(subset="iaps_id")
        .set_index("iaps_id")
    )
    ordered = unique.loc[stim_order["iaps_id"]]
    assert ordered.shape[0] == 60
    assert not ordered.isna().any().any()
    va_raw = ordered[["Mean_Arousal", "Mean_Valence"]].to_numpy(dtype=np.float64)
    rdm = pairwise_distances(va_raw, metric="euclidean")

    # The historical behavioral RDM was built from these same raw dimensions.
    historical_path = os.path.join(c.BASE_DIR, "RDM", "rdm_behavioral.csv")
    if os.path.exists(historical_path):
        historical = load_rdm_csv(historical_path)
        assert np.allclose(rdm, historical, atol=1e-10), (
            "newly built raw-VA RDM does not match RDM/rdm_behavioral.csv"
        )

    out_path = os.path.join(c.CODES_DIR, "rdm_va_raw.csv")
    c.save_rdm_csv(rdm, out_path)
    return rdm


def load_eval_csv(path: str) -> np.ndarray:
    scores = pd.read_csv(path, index_col=0).to_numpy(dtype=np.float64)
    assert scores.shape[0] == 20, f"expected 20 subjects in {path}, got {scores.shape}"
    assert np.all(np.isfinite(scores)), f"non-finite evaluation scores in {path}"
    return scores


def load_cache_and_metadata() -> tuple[dict, np.ndarray, tuple[int, int, int]]:
    print(f"Loading neural RDM cache: {c.SL_CACHE_PATH}")
    with open(c.SL_CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    voxel_index = np.asarray(cache["voxel_index"], dtype=np.int64)
    mask_shape = tuple(int(v) for v in cache["mask_shape"])
    np.savez_compressed(METADATA_PATH, voxel_index=voxel_index, mask_shape=mask_shape)
    return cache, voxel_index, mask_shape


def evaluate_raw_va_if_needed(cache: dict, raw_rdm: np.ndarray) -> np.ndarray:
    if os.path.exists(EVAL_PATHS["va_raw"]):
        print(f"Loading existing raw-VA score table: {EVAL_PATHS['va_raw']}")
        return load_eval_csv(EVAL_PATHS["va_raw"])

    print("Computing raw-VA Spearman scores against the cached neural RDMs...")
    scores, returned_index, returned_shape = c.evaluate_model_against_cache(
        cache, c.upper_tri(raw_rdm), "VA (raw Euclidean)"
    )
    assert np.array_equal(returned_index, cache["voxel_index"])
    assert tuple(returned_shape) == tuple(cache["mask_shape"])
    pd.DataFrame(scores).to_csv(EVAL_PATHS["va_raw"])
    print(f"Saved raw-VA score table -> {EVAL_PATHS['va_raw']}")
    return np.asarray(scores, dtype=np.float64)


def rank_standardize(vector: np.ndarray) -> np.ndarray:
    ranked = rankdata(np.asarray(vector, dtype=np.float64), method="average")
    return (ranked - ranked.mean()) / ranked.std(ddof=1)


def predictor_geometry(rdms: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rank-z predictor matrix, correlation matrix, and its inverse."""
    x = np.column_stack([rank_standardize(c.upper_tri(rdms[name])) for name in PREDICTOR_NAMES])
    correlation = np.corrcoef(x, rowvar=False)
    inverse = np.linalg.inv(correlation)
    return x, correlation, inverse


def standardized_betas(
    score_arrays: dict[str, np.ndarray], inverse_predictor_correlation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute beta = inv(R_xx) r_xy using the exact single-model Spearman scores.

    Returns betas (subjects, centers, predictors) and full-model R^2
    (subjects, centers).
    """
    r_xy = np.stack([score_arrays[name] for name in PREDICTOR_NAMES], axis=-1)
    betas = r_xy @ inverse_predictor_correlation
    full_r2 = np.sum(r_xy * betas, axis=-1)
    return betas, full_r2


def validate_beta_identity(
    cache: dict,
    x_rank_z: np.ndarray,
    betas: np.ndarray,
    subject_center_pairs: tuple[tuple[int, int], ...] = ((0, 0), (0, 1000), (9, 20000), (19, 50000)),
) -> float:
    """Check beta=inv(R)r against direct rank-OLS at representative searchlights."""
    max_error = 0.0
    for subject, center in subject_center_pairs:
        y_rank_z = rank_standardize(cache["dissimilarities"][subject, center])
        direct_beta, *_ = np.linalg.lstsq(x_rank_z, y_rank_z, rcond=None)
        error = float(np.max(np.abs(direct_beta - betas[subject, center])))
        max_error = max(max_error, error)
    if max_error > 1e-6:
        raise AssertionError(f"beta identity validation failed; max abs error={max_error:.3g}")
    return max_error


def one_sample_t(beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_stat, p_two_sided = ttest_1samp(beta, popmean=0, axis=0)
    return np.nan_to_num(t_stat, nan=0.0), np.nan_to_num(p_two_sided, nan=1.0)


def max_t_fwe_pvalues(
    beta: np.ndarray,
    t_observed: np.ndarray,
    signs: np.ndarray,
    batch_size: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided synchronized sign-flip max-T correction across all searchlights."""
    beta = np.asarray(beta, dtype=np.float64)
    n_subjects = beta.shape[0]
    sum_squares = np.sum(beta * beta, axis=0)
    max_statistics = np.empty(signs.shape[0], dtype=np.float64)

    for start in range(0, signs.shape[0], batch_size):
        stop = min(start + batch_size, signs.shape[0])
        signed_sums = signs[start:stop] @ beta
        centered_ss = sum_squares[None, :] - (signed_sums * signed_sums) / n_subjects
        variance = np.maximum(centered_ss / (n_subjects - 1), np.finfo(float).tiny)
        t_perm = signed_sums / np.sqrt(n_subjects * variance)
        max_statistics[start:stop] = np.max(t_perm, axis=1)

    sorted_max = np.sort(max_statistics)
    first_ge = np.searchsorted(sorted_max, t_observed, side="left")
    count_ge = len(sorted_max) - first_ge
    p_fwe = (count_ge + 1.0) / (len(sorted_max) + 1.0)
    return p_fwe, max_statistics


def cluster_filter(mask: np.ndarray, minimum_size: int) -> np.ndarray:
    """Keep face-connected clusters containing at least minimum_size voxels."""
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, n_labels = ndimage.label(mask, structure=structure)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    keep_labels = np.flatnonzero(counts >= minimum_size)
    keep_labels = keep_labels[keep_labels != 0]
    return np.isin(labels, keep_labels)


def center_values_to_volume(
    values: np.ndarray,
    voxel_index: np.ndarray,
    mask_shape: tuple[int, int, int],
) -> np.ndarray:
    full = np.zeros(int(np.prod(mask_shape)), dtype=np.float32)
    full[voxel_index] = np.asarray(values, dtype=np.float32)
    return full.reshape(mask_shape)


def save_volume(data: np.ndarray, out_path: str, reference: nib.spatialimages.SpatialImage) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), reference.affine, header)
    nib.save(img, out_path)


def save_group_maps(
    model_tag: str,
    predictor: str,
    beta: np.ndarray,
    voxel_index: np.ndarray,
    mask_shape: tuple[int, int, int],
    reference: nib.spatialimages.SpatialImage,
    signs: np.ndarray,
) -> dict:
    t_stat, p_two_sided = one_sample_t(beta)
    p_fwe, max_statistics = max_t_fwe_pvalues(beta, t_stat, signs)

    stem = f"mrsa_clip_gist_{model_tag}_beta_{predictor}"
    np.save(os.path.join(c.CODES_DIR, f"tstat_{stem}.npy"), t_stat)
    np.save(os.path.join(c.CODES_DIR, f"pvalue_two_sided_{stem}.npy"), p_two_sided)
    np.save(os.path.join(c.CODES_DIR, f"pvalue_maxT_fwe_{stem}.npy"), p_fwe)
    np.save(os.path.join(c.CODES_DIR, f"null_maxT_{stem}.npy"), max_statistics)

    os.makedirs(c.FINAL_RESULTS_DIR, exist_ok=True)
    unthresholded_path = os.path.join(c.FINAL_RESULTS_DIR, f"tmap_{stem}_unthresholded.nii.gz")
    save_volume(center_values_to_volume(t_stat, voxel_index, mask_shape), unthresholded_path, reference)

    fwe_mask = (p_fwe < 0.05) & (t_stat > 0)
    fwe_values = np.where(fwe_mask, t_stat, 0)
    fwe_path = os.path.join(c.FINAL_RESULTS_DIR, f"tmap_{stem}_maxT_fwe05.nii.gz")
    save_volume(center_values_to_volume(fwe_values, voxel_index, mask_shape), fwe_path, reference)

    result = {
        "unthresholded_path": unthresholded_path,
        "maxT_fwe05_path": fwe_path,
        "maxT_fwe05_voxels": int(fwe_mask.sum()),
        "maxT_95th_percentile": float(np.quantile(max_statistics, 0.95)),
        "mean_beta": float(np.mean(beta)),
        "mean_t": float(np.mean(t_stat)),
        "max_t": float(np.max(t_stat)),
    }

    for fdr_tag, alpha in FDR_ALPHAS.items():
        rejected, corrected_p = fdrcorrection(p_two_sided, alpha=alpha, method="indep")
        center_mask = rejected & (t_stat > 0)
        spatial_mask = center_values_to_volume(center_mask.astype(np.float32), voxel_index, mask_shape) > 0
        clustered_spatial_mask = cluster_filter(spatial_mask, CLUSTER_THRESHOLD)
        clustered_center_mask = clustered_spatial_mask.reshape(-1)[voxel_index]
        values = np.where(clustered_center_mask, t_stat, 0)
        out_path = os.path.join(
            c.FINAL_RESULTS_DIR,
            f"tmap_{stem}_{fdr_tag}_cluster{CLUSTER_THRESHOLD}.nii.gz",
        )
        save_volume(center_values_to_volume(values, voxel_index, mask_shape), out_path, reference)
        np.save(os.path.join(c.CODES_DIR, f"pvalue_bh_{stem}_{fdr_tag}.npy"), corrected_p)
        result[f"{fdr_tag}_precluster_voxels"] = int(center_mask.sum())
        result[f"{fdr_tag}_cluster{CLUSTER_THRESHOLD}_voxels"] = int(clustered_center_mask.sum())
        result[f"{fdr_tag}_cluster{CLUSTER_THRESHOLD}_path"] = out_path

    return result


def main() -> None:
    raw_va_rdm = build_raw_va_rdm()
    model_rdms_common = {
        "clip": load_rdm_csv(RDM_PATHS["clip"]),
        "gist": load_rdm_csv(RDM_PATHS["gist"]),
    }
    zscore_va_rdm = load_rdm_csv(RDM_PATHS["va_zscore"])

    cache, voxel_index, mask_shape = load_cache_and_metadata()
    raw_va_scores = evaluate_raw_va_if_needed(cache, raw_va_rdm)
    common_scores = {
        "clip": load_eval_csv(EVAL_PATHS["clip"]),
        "gist": load_eval_csv(EVAL_PATHS["gist"]),
    }
    zscore_va_scores = load_eval_csv(EVAL_PATHS["va_zscore"])

    n_centers = len(voxel_index)
    for name, scores in {**common_scores, "va_raw": raw_va_scores, "va_zscore": zscore_va_scores}.items():
        assert scores.shape == (20, n_centers), (
            f"{name} score shape {scores.shape} does not match 20 subjects x {n_centers} centers"
        )

    rng = np.random.default_rng(PERMUTATION_SEED)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(N_PERMUTATIONS, 20), replace=True)
    reference = c.get_tmp_img()

    summary = {
        "method": {
            "dependent_rdm": "neural correlation-distance RDM",
            "regression": "rank-standardized OLS (multiple-regression Spearman RSA)",
            "predictors": list(PREDICTOR_NAMES),
            "primary_inference": "one-sided synchronized subject sign-flip max-T voxelwise FWE",
            "n_permutations": N_PERMUTATIONS,
            "permutation_seed": PERMUTATION_SEED,
            "secondary_inference": "two-sided parametric p, BH-FDR, retain positive t, 30-voxel extent filter",
            "n_subjects": 20,
            "n_searchlight_centers": n_centers,
        },
        "models": {},
    }

    model_inputs = {
        "va_raw": (raw_va_rdm, raw_va_scores),
        "va_zscore": (zscore_va_rdm, zscore_va_scores),
    }

    for model_tag, (va_rdm, va_scores) in model_inputs.items():
        print(f"\nRunning multiple-regression RSA: {model_tag}")
        rdms = {**model_rdms_common, "va": va_rdm}
        scores = {**common_scores, "va": va_scores}
        x_rank_z, predictor_correlation, inverse_correlation = predictor_geometry(rdms)
        betas, full_r2 = standardized_betas(scores, inverse_correlation)

        validation_error = validate_beta_identity(cache, x_rank_z, betas)
        np.save(os.path.join(c.CODES_DIR, f"betas_mrsa_clip_gist_{model_tag}.npy"), betas.astype(np.float32))
        np.save(os.path.join(c.CODES_DIR, f"r2_mrsa_clip_gist_{model_tag}.npy"), full_r2.astype(np.float32))

        mean_r2_volume = center_values_to_volume(np.mean(full_r2, axis=0), voxel_index, mask_shape)
        mean_r2_path = os.path.join(c.FINAL_RESULTS_DIR, f"mean_r2_mrsa_clip_gist_{model_tag}.nii.gz")
        save_volume(mean_r2_volume, mean_r2_path, reference)

        model_summary = {
            "predictor_spearman_correlation": predictor_correlation.tolist(),
            "predictor_order": list(PREDICTOR_NAMES),
            "variance_inflation_factor": np.diag(inverse_correlation).tolist(),
            "condition_number": float(np.linalg.cond(predictor_correlation)),
            "direct_ols_validation_max_abs_error": validation_error,
            "mean_full_r2": float(np.mean(full_r2)),
            "mean_r2_path": mean_r2_path,
            "predictor_maps": {},
        }

        for predictor_index, predictor in enumerate(PREDICTOR_NAMES):
            print(f"Group inference for {model_tag}, unique {predictor} beta...")
            map_summary = save_group_maps(
                model_tag,
                predictor,
                betas[:, :, predictor_index],
                voxel_index,
                mask_shape,
                reference,
                signs,
            )
            model_summary["predictor_maps"][predictor] = map_summary
            print(
                f"  max-T FWE p<0.05: {map_summary['maxT_fwe05_voxels']} voxels; "
                f"FDR q<0.01 cluster30: {map_summary['fdr01_cluster30_voxels']} voxels; "
                f"FDR q<0.001 cluster30: {map_summary['fdr001_cluster30_voxels']} voxels"
            )

        summary["models"][model_tag] = model_summary

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved analysis summary -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
