"""
Stage 2: Valence/Arousal model RDM, fixed.

Original problem (see codes/README.md): rdm_behavioral.csv / rdm_behavioral_normed.csv
were built from *raw* Mean_Valence/Mean_Arousal (no z-scoring) before taking Euclidean
distance. Valence spans nearly the full 1-9 IAPS scale while arousal is compressed
toward the mid-range, so the raw-Euclidean RDM is valence-dominated rather than a
genuine joint VA model.

Fix: z-score valence and arousal independently across the 60 stimuli, THEN take
Euclidean distance on the two z-scored dimensions.

Requires codes/build_neural_searchlight_rdms.py to have been run already
(reads codes/SL_RDM_allsubjects.pkl).
"""
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics import pairwise_distances

import common as c


def build_va_rdm():
    stim_order = c.load_stim_order()  # canonical 60-stim order, columns: iaps_id, emotion_type

    conditions = pd.read_csv(c.BEHAVIORAL_CONDITIONS_PATH)
    unique = (
        conditions[["iaps_id", "Mean_Valence", "Mean_Arousal"]]
        .drop_duplicates(subset="iaps_id")
        .set_index("iaps_id")
    )

    # Reindex to the canonical 60-stimulus order so this RDM's row/column order matches
    # allsub_avg.npy's condition order and every other model RDM built in this folder.
    ordered = unique.loc[stim_order["iaps_id"]]
    assert ordered.shape[0] == 60, f"expected 60 stimuli, got {ordered.shape[0]}"
    assert not ordered.isna().any().any(), "missing valence/arousal for some stimulus"

    va_z = np.column_stack([
        zscore(ordered["Mean_Arousal"].to_numpy(), ddof=1),
        zscore(ordered["Mean_Valence"].to_numpy(), ddof=1),
    ])
    rdm = pairwise_distances(va_z, metric="euclidean")
    return rdm


def main():
    rdm = build_va_rdm()
    rdm_path = c.CODES_DIR + "/rdm_va_zscore.csv"
    c.save_rdm_csv(rdm, rdm_path)

    cache = c.load_neural_rdm_cache()
    model_vec = c.upper_tri(rdm)
    eval_score_array, voxel_index, mask_shape = c.evaluate_model_against_cache(
        cache, model_vec, "VA (z-scored Euclidean)"
    )

    pd.DataFrame(eval_score_array).to_csv(c.CODES_DIR + "/eval_score_va_zscore.csv")
    c.group_stats_and_save(
        eval_score_array,
        voxel_index,
        mask_shape,
        out_prefix="va_zscore_euclidean",
        cluster_threshold=30,
        fdr_tags=("fdr01",),
        output_suffix="_cluster30",
    )


if __name__ == "__main__":
    main()
