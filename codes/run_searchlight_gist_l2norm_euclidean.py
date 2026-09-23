"""
Stage 2: GIST model RDM, L2-normalized-Euclidean variant.

Original problem (see codes/README.md): gist_rdm_euc.csv was built from *raw* GIST
descriptors (unnormalized oriented energy) straight into Euclidean distance. Images
that are simply higher-contrast or busier have larger descriptor norms, so the
unnormalized-Euclidean RDM partly indexes overall image energy/contrast rather than
spatial-envelope structure.

Fix: L2-normalize each image's GIST descriptor to unit norm, THEN take Euclidean
distance -- a structural comparison, matching the Oliva & Torralba convention while
removing the contrast/energy confound.

GIST descriptors are read from IAPS_Gist_RDM/gist/{iaps_id}.jpg.txt, in the same
canonical IAPS_60_pnu.csv order used throughout this folder -- this matches exactly
how IAPS_Gist_RDM/searchlight_fmri_gist.ipynb reads them (stim_order_txt built from
the same IAPS_60_pnu.csv), so ordering is consistent with the neural data and the
other two model RDMs.

Requires codes/build_neural_searchlight_rdms.py to have been run already
(reads codes/SL_RDM_allsubjects.pkl).
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

import common as c


def load_gist_descriptors():
    stim_order = c.load_stim_order()
    paths = [os.path.join(c.GIST_DIR, f"{iaps_id}.jpg.txt") for iaps_id in stim_order["iaps_id"]]
    missing = [p for p in paths if not os.path.exists(p)]
    assert not missing, f"missing GIST descriptor files: {missing}"

    gist = np.stack([np.loadtxt(p) for p in paths], axis=0)  # (60, n_features)
    assert gist.shape[0] == 60, f"expected 60 GIST descriptors, got {gist.shape[0]}"
    return gist


def main():
    gist = load_gist_descriptors()
    gist_l2 = normalize(gist, norm="l2", axis=1)
    rdm = pairwise_distances(gist_l2, metric="euclidean")

    rdm_path = c.CODES_DIR + "/rdm_gist_l2norm_euclidean.csv"
    c.save_rdm_csv(rdm, rdm_path)

    cache = c.load_neural_rdm_cache()
    model_vec = c.upper_tri(rdm)
    eval_score_array, voxel_index, mask_shape = c.evaluate_model_against_cache(
        cache, model_vec, "GIST (L2-normalized Euclidean)"
    )

    pd.DataFrame(eval_score_array).to_csv(c.CODES_DIR + "/eval_score_gist_l2norm_euclidean.csv")
    c.group_stats_and_save(eval_score_array, voxel_index, mask_shape, out_prefix="gist_l2norm_euclidean")


if __name__ == "__main__":
    main()
