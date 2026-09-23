"""
Stage 2: CLIP-ViT model RDM, L2-normalized-Euclidean variant (comparison to cosine).

Same clip_image_embeddigs.pkl as run_searchlight_clip_cosine.py (CLIP ViT-B/32 image
embeddings, verified provenance in that script's docstring). Here each embedding row is
L2-normalized to unit norm first, then Euclidean distance is taken.

This is run as an explicit empirical check against run_searchlight_clip_cosine.py, not
because Euclidean is the recommended metric for CLIP: for L2-normalized vectors,
squared Euclidean distance is a strictly monotonic function of cosine distance
(d^2 = 2 - 2*cos), so under a rank-based second-level metric (Spearman, used here) the
two should give effectively identical searchlight results. Running both makes that
verifiable rather than assumed.

Requires codes/build_neural_searchlight_rdms.py to have been run already
(reads codes/SL_RDM_allsubjects.pkl).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

import common as c
from run_searchlight_clip_cosine import load_clip_image_embeddings


def main():
    emb = load_clip_image_embeddings()
    emb_l2 = normalize(emb, norm="l2", axis=1)
    rdm = pairwise_distances(emb_l2, metric="euclidean")

    rdm_path = c.CODES_DIR + "/rdm_clip_euclidean_l2norm.csv"
    c.save_rdm_csv(rdm, rdm_path)

    cache = c.load_neural_rdm_cache()
    model_vec = c.upper_tri(rdm)
    eval_score_array, voxel_index, mask_shape = c.evaluate_model_against_cache(
        cache, model_vec, "CLIP-ViT (Euclidean, L2-normalized)"
    )

    pd.DataFrame(eval_score_array).to_csv(c.CODES_DIR + "/eval_score_clip_euclidean_l2norm.csv")
    c.group_stats_and_save(eval_score_array, voxel_index, mask_shape, out_prefix="clip_euclidean_l2norm")


if __name__ == "__main__":
    main()
