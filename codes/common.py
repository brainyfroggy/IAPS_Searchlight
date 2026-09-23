"""
Shared paths, constants, and helper functions for the revised-RDM searchlight
scripts in this folder (build_neural_searchlight_rdms.py + run_searchlight_*.py).

Kept as one shared module (rather than duplicated in each script) so the group-level
statistics routine only exists in one place -- see codes/README.md for why this exists
and what problem each script fixes.
"""
import os
import pickle

import numpy as np
import pandas as pd
import nibabel as nib
import nilearn.image as nlimg
from nilearn.image import new_img_like
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection
from rsatoolbox.rdm.rdms import RDMs
from rsatoolbox.model import ModelFixed
from rsatoolbox.inference import eval_fixed
from rsatoolbox.util.searchlight import evaluate_models_searchlight

CODES_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODES_DIR)  # IAPS_Searchlight/
PROJECTS_DIR = os.path.dirname(BASE_DIR)  # .../projects/

MASK_PATH = os.path.join(BASE_DIR, "mask.npy")
ALLSUB_AVG_PATH = os.path.join(BASE_DIR, "allsub_avg.npy")
# Canonical 60-stimulus order (Pl 1-20, Nt 1-20, Up 1-20 by stim_order) -- this is the
# SAME file already used to order both the CLIP image list and the GIST text-file list
# in the existing notebooks (searchlight_ai_embeddings_models.ipynb,
# IAPS_Gist_RDM/searchlight_fmri_gist.ipynb), and matches the condition order baked
# into allsub_avg.npy. Every model RDM built in this folder is aligned to this order.
STIM_ORDER_PATH = os.path.join(BASE_DIR, "IAPS_60_pnu.csv")
BEHAVIORAL_CONDITIONS_PATH = os.path.join(PROJECTS_DIR, "IAPS_behavioral_RDM", "fmri_conditions_trial_ordered.csv")
CLIP_IMAGE_EMBEDDINGS_PATH = os.path.join(BASE_DIR, "clip_image_embeddigs.pkl")
GIST_DIR = os.path.join(PROJECTS_DIR, "IAPS_Gist_RDM", "gist")
TMP_IMG_PATH = os.path.join(PROJECTS_DIR, "IAPS_fMRI_RSA", "fMRI_singletrial_betas", "nifti", "Nt1.img")

SL_CACHE_PATH = os.path.join(CODES_DIR, "SL_RDM_allsubjects.pkl")
FINAL_RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "tBrainmap", "final_results2")

RADIUS = 5
THRESHOLD = 0.5
NEURAL_RDM_METHOD = "correlation"
SECOND_LEVEL_METHOD = "spearman"
N_JOBS = 6

# FDR q-value thresholds this run reports at, and the voxel-cluster minimum applied
# after thresholding -- both match the convention already used elsewhere in this
# project (e.g. searchlight_behavioral_final2_fdr05.ipynb).
FDR_ALPHAS = {"fdr05": 0.05, "fdr01": 0.01, "fdr001": 0.001}
CLUSTER_THRESHOLD = 10


def upper_tri(rdm):
    """Upper-triangular vector (excluding diagonal) of a square RDM."""
    m = rdm.shape[0]
    r, c = np.triu_indices(m, 1)
    return rdm[r, c]


def load_mask():
    return np.load(MASK_PATH)


def load_allsub_avg():
    """(n_subjects, n_conditions, n_voxels) array, condition order = IAPS_60_pnu.csv order."""
    return np.load(ALLSUB_AVG_PATH, allow_pickle=True)


def load_stim_order():
    """DataFrame with columns [iaps_id, emotion_type]; row order IS the canonical order."""
    return pd.read_csv(STIM_ORDER_PATH)


def get_tmp_img():
    return nib.load(TMP_IMG_PATH)


def save_rdm_csv(rdm, path):
    pd.DataFrame(rdm).to_csv(path)
    print(f"Saved RDM {rdm.shape} -> {path}")


def load_neural_rdm_cache():
    """Load the Stage-1 cache produced by build_neural_searchlight_rdms.py."""
    if not os.path.exists(SL_CACHE_PATH):
        raise FileNotFoundError(
            f"{SL_CACHE_PATH} not found -- run build_neural_searchlight_rdms.py first "
            "(Stage 1 must complete before any run_searchlight_*.py script)."
        )
    with open(SL_CACHE_PATH, "rb") as f:
        return pickle.load(f)


def evaluate_model_against_cache(cache, model_rdm_vector, model_name):
    """
    Re-hydrate each subject's cached neural RDM into an rsatoolbox RDMs object and
    evaluate it against a fixed model RDM with evaluate_models_searchlight(method='spearman'),
    exactly as searchlight_behavioral_final.ipynb does -- this reuses rsatoolbox's own
    comparison code (rather than re-implementing Spearman-vs-RDM by hand) so the
    per-center evaluation logic is identical to the rest of this project.

    Args:
        cache: dict from load_neural_rdm_cache()
        model_rdm_vector: upper-triangular model RDM vector (from common.upper_tri)
        model_name: label used only for the ModelFixed object / progress messages

    Returns:
        (eval_score_array, voxel_index, mask_shape):
            eval_score_array: (n_subjects, n_centers) Spearman correlations
            voxel_index: (n_centers,) flat voxel index per center
            mask_shape: (x, y, z)
    """
    model = ModelFixed(model_name, model_rdm_vector)
    dissimilarities = cache["dissimilarities"]
    voxel_index = cache["voxel_index"]
    n_subjects = dissimilarities.shape[0]

    eval_score_list = []
    for s in range(n_subjects):
        print(f"[{model_name}] evaluating subject {s + 1}/{n_subjects}...")
        SL_RDM = RDMs(
            dissimilarities[s],
            rdm_descriptors={"voxel_index": voxel_index},
            dissimilarity_measure=cache["method"],
        )
        eval_results = evaluate_models_searchlight(
            SL_RDM, model, eval_fixed, method=SECOND_LEVEL_METHOD, n_jobs=N_JOBS
        )
        eval_score_list.append([float(e.evaluations) for e in eval_results])

    return np.array(eval_score_list), voxel_index, cache["mask_shape"]


def group_stats_and_save(
    eval_score_array,
    voxel_index,
    mask_shape,
    out_prefix,
    cluster_threshold=CLUSTER_THRESHOLD,
    fdr_tags=None,
    output_suffix="",
):
    """
    Fisher-z transform per-subject Spearman correlations, one-sample t-test across
    subjects, then FDR-correct independently at each of FDR_ALPHAS and save a
    cluster-thresholded, positive-tail-only t-map per alpha.

    This is the exact statistical recipe already used in searchlight_behavioral_final.ipynb
    (Fisher z -> ttest_1samp -> fdrcorrection -> cluster threshold -> positive tail only),
    just parameterized over multiple FDR levels instead of one.

    Args:
        eval_score_array: (n_subjects, n_centers) Spearman correlations per searchlight center
        voxel_index: (n_centers,) flat-voxel index for each center (same order as columns above)
        mask_shape: (x, y, z) shape of the whole-brain mask used for the searchlight
        out_prefix: e.g. 'va_zscore_euclidean' -> tmap_va_zscore_euclidean_fdr05.nii.gz etc.
        cluster_threshold: minimum voxel cluster size passed to nilearn.image.threshold_img.
        fdr_tags: optional iterable of FDR tags to save; defaults to every tag in FDR_ALPHAS.
        output_suffix: optional suffix before .nii.gz, e.g. '_cluster30'.

    Returns:
        dict mapping fdr tag ('fdr05'/'fdr01'/'fdr001') -> saved .nii.gz path
    """
    os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)

    eval_score_array = np.asarray(eval_score_array, dtype=float)
    epsilon = 1e-10
    eval_score_array = np.clip(eval_score_array, -1 + epsilon, 1 - epsilon)
    fisher_z = np.arctanh(eval_score_array)

    t_stat, p_value = ttest_1samp(fisher_z, popmean=0, axis=0)

    np.save(os.path.join(CODES_DIR, f"tstat_{out_prefix}.npy"), t_stat)
    np.save(os.path.join(CODES_DIR, f"pvalue_{out_prefix}.npy"), p_value)

    tmp_img = get_tmp_img()
    x, y, z = mask_shape
    voxel_index = np.asarray(voxel_index)

    selected_alphas = FDR_ALPHAS if fdr_tags is None else {tag: FDR_ALPHAS[tag] for tag in fdr_tags}

    saved_paths = {}
    for tag, alpha in selected_alphas.items():
        _, corrected_p = fdrcorrection(p_value, alpha=alpha, method="indep", is_sorted=False)
        sig = np.where(corrected_p < alpha)[0]

        t_full = np.full(x * y * z, np.nan)
        t_full[voxel_index[sig]] = t_stat[sig]
        t_3d = t_full.reshape(x, y, z)
        t_3d[t_3d < 0] = np.nan  # positive tail only, matching existing convention

        img = new_img_like(tmp_img, t_3d)
        img_clustered = nlimg.threshold_img(img, 0, cluster_threshold=cluster_threshold)

        out_path = os.path.join(FINAL_RESULTS_DIR, f"tmap_{out_prefix}_{tag}{output_suffix}.nii.gz")
        nib.save(img_clustered, out_path)
        saved_paths[tag] = out_path
        print(
            f"[{out_prefix}] {tag} (q<{alpha}, cluster>={cluster_threshold} voxels): "
            f"{len(sig)} significant voxels -> {out_path}"
        )

    return saved_paths
