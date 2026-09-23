"""
Stage 2: OpenAI CLIP ResNet50 image model RDMs.

This script adds the ResNet-backbone CLIP model used in Wang et al. (2023):
OpenAI CLIP with a ResNet50 image encoder (`clip.load("RN50")`). The extracted
image embedding has 1024 dimensions, matching the paper's model-description table.

The neural side is unchanged: this reads the existing Stage-1 searchlight RDM cache
(`SL_RDM_allsubjects.pkl`) and uses the same RSA/group-statistics pipeline as the
other scripts in this folder.

Outputs in codes/:
- resnetclip_rn50_image_embeddings.pkl
- rdm_resnetclip_cosine.csv
- rdm_resnetclip_euclidean_l2norm.csv
- eval_score_resnetclip_cosine.csv
- eval_score_resnetclip_euclidean_l2norm.csv
- tstat_resnetclip_*.npy and pvalue_resnetclip_*.npy

FDR q<0.01 maps are saved in outputs/tBrainmap/final_results2/:
- tmap_resnetclip_cosine_fdr01.nii.gz
- tmap_resnetclip_euclidean_l2norm_fdr01.nii.gz
"""
import os
import pickle

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

import common as c


MODEL_NAME = "RN50"
EMBEDDINGS_PATH = os.path.join(c.BASE_DIR, "resnetclip_rn50_image_embeddings.pkl")
RAW_IAPS_DIR = os.path.join(c.BASE_DIR, "data", "raw_iaps")


def load_iaps_image_paths():
    stim_order = c.load_stim_order()
    paths = [
        os.path.join(RAW_IAPS_DIR, f"{int(iaps_id)}.jpg")
        for iaps_id in stim_order["iaps_id"]
    ]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing IAPS image files: {missing[:5]}")
    return paths


def extract_or_load_resnetclip_embeddings(force=False):
    if os.path.exists(EMBEDDINGS_PATH) and not force:
        with open(EMBEDDINGS_PATH, "rb") as f:
            emb = pickle.load(f)
        emb = np.asarray(emb, dtype=np.float64)
        assert emb.shape == (60, 1024), f"expected (60, 1024), got {emb.shape}"
        print(f"Loaded existing ResNet-CLIP RN50 embeddings {emb.shape} -> {EMBEDDINGS_PATH}")
        return emb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()

    features = []
    for image_path in load_iaps_image_paths():
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        features.append(image_features.detach().cpu().numpy().squeeze(0))

    emb = np.asarray(features, dtype=np.float64)
    assert emb.shape == (60, 1024), f"expected (60, 1024), got {emb.shape}"

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(emb, f)
    print(f"Saved ResNet-CLIP RN50 embeddings {emb.shape} -> {EMBEDDINGS_PATH}")
    return emb


def run_one_model(cache, rdm, out_prefix, model_label):
    c.save_rdm_csv(rdm, os.path.join(c.CODES_DIR, f"rdm_{out_prefix}.csv"))
    model_vec = c.upper_tri(rdm)
    eval_score_array, voxel_index, mask_shape = c.evaluate_model_against_cache(
        cache, model_vec, model_label
    )
    pd.DataFrame(eval_score_array).to_csv(os.path.join(c.CODES_DIR, f"eval_score_{out_prefix}.csv"))
    return c.group_stats_and_save(
        eval_score_array,
        voxel_index,
        mask_shape,
        out_prefix=out_prefix,
        fdr_tags=["fdr01"],
    )


def main():
    emb = extract_or_load_resnetclip_embeddings()
    rdm_cosine = pairwise_distances(emb, metric="cosine")
    emb_l2 = normalize(emb, norm="l2", axis=1)
    rdm_euclidean_l2norm = pairwise_distances(emb_l2, metric="euclidean")

    rho = spearmanr(c.upper_tri(rdm_cosine), c.upper_tri(rdm_euclidean_l2norm)).statistic
    print(f"Spearman rho(cosine RDM, L2-normalized Euclidean RDM) = {rho:.12f}")

    cache = c.load_neural_rdm_cache()
    run_one_model(cache, rdm_cosine, "resnetclip_cosine", "CLIP-RN50 (cosine)")
    run_one_model(
        cache,
        rdm_euclidean_l2norm,
        "resnetclip_euclidean_l2norm",
        "CLIP-RN50 (Euclidean, L2-normalized)",
    )


if __name__ == "__main__":
    main()
