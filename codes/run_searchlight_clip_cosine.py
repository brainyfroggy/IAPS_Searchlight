"""
Stage 2: CLIP-ViT model RDM, cosine-distance variant.

Original problem (see codes/README.md): rdm_clipvit.csv was built with
pairwise_distances(clip_image_embeddigs, metric='correlation'), which mean-centers
each embedding vector -- not the geometry CLIP's contrastive objective was trained
under (cosine similarity on the raw/L2-normalized embedding).

Fix: cosine distance (1 - cosine similarity) directly on clip_image_embeddigs.pkl.

clip_image_embeddigs.pkl provenance (verified by reading searchlight_ai_embeddings_models.ipynb):
OpenAI CLIP ViT-B/32 (`clip.load("ViT-B/32")`), `model.encode_image(image)` output,
shape (60, 512) -- genuine CLIP-ViT *image* embeddings, extracted in the same
IAPS_60_pnu.csv stimulus order used throughout this folder. This is NOT
clip_text_embeddigs.pkl (text encoder) and NOT the separate "pure_vit_embeddings"
(a plain ImageNet ViT unrelated to CLIP) also present in that notebook.

Companion script run_searchlight_clip_euclidean_l2norm.py runs Euclidean distance on
L2-normalized embeddings as a direct empirical comparison -- the two are expected to
give near-identical results under Spearman since Euclidean-on-L2-normed vectors is a
monotonic function of cosine distance.

Requires codes/build_neural_searchlight_rdms.py to have been run already
(reads codes/SL_RDM_allsubjects.pkl).
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

import common as c


def load_clip_image_embeddings():
    with open(c.CLIP_IMAGE_EMBEDDINGS_PATH, "rb") as f:
        emb = pickle.load(f)
    emb = np.asarray(emb, dtype=np.float64)
    assert emb.shape[0] == 60, f"expected 60 CLIP image embeddings, got {emb.shape[0]}"
    return emb


def main():
    emb = load_clip_image_embeddings()
    rdm = pairwise_distances(emb, metric="cosine")

    rdm_path = c.CODES_DIR + "/rdm_clip_cosine.csv"
    c.save_rdm_csv(rdm, rdm_path)

    cache = c.load_neural_rdm_cache()
    model_vec = c.upper_tri(rdm)
    eval_score_array, voxel_index, mask_shape = c.evaluate_model_against_cache(
        cache, model_vec, "CLIP-ViT (cosine)"
    )

    pd.DataFrame(eval_score_array).to_csv(c.CODES_DIR + "/eval_score_clip_cosine.csv")
    c.group_stats_and_save(eval_score_array, voxel_index, mask_shape, out_prefix="clip_cosine")


if __name__ == "__main__":
    main()
