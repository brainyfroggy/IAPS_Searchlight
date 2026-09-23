"""
Stage 1 (run once): compute the per-subject, per-searchlight-center neural RDM
(correlation distance across voxel patterns -- unchanged from searchlight_behavioral_final.ipynb)
and cache it to disk.

This step is identical regardless of which model RDM (VA / CLIP / GIST) will later be
compared against it, so it is split out from the 4 run_searchlight_*.py scripts to avoid
recomputing the same ~20-subject searchlight pass 4 times. See codes/README.md for the
full reasoning.

Run this once, then the 4 run_searchlight_*.py scripts can be launched in parallel --
they only load the cache this script produces.
"""
import pickle
import time

import numpy as np
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs

import common as c


def main():
    mask = c.load_mask()
    print(f"Mask shape: {mask.shape}, voxels: {int(mask.sum())}")

    print(f"Building searchlight centers/neighbors (radius={c.RADIUS}, threshold={c.THRESHOLD})...")
    centers, neighbors = get_volume_searchlight(mask, radius=c.RADIUS, threshold=c.THRESHOLD)
    print(f"Found {len(centers)} searchlight centers")

    allsub_avg = c.load_allsub_avg()
    print(f"allsub_avg shape: {allsub_avg.shape}")
    n_subjects, n_conditions = allsub_avg.shape[0], allsub_avg.shape[1]
    image_value = np.arange(n_conditions)

    dissimilarities = []
    voxel_index = None
    t0 = time.time()
    for s in range(n_subjects):
        print(f"Subject {s + 1}/{n_subjects} ({time.time() - t0:.0f}s elapsed)...")
        subj_data = np.nan_to_num(allsub_avg[s])
        SL_RDM = get_searchlight_RDMs(
            subj_data, centers, neighbors, image_value, method=c.NEURAL_RDM_METHOD
        )
        # float32 keeps the cache to a manageable size (n_subjects x n_centers x n_pairs)
        dissimilarities.append(SL_RDM.dissimilarities.astype(np.float32))
        if voxel_index is None:
            voxel_index = np.asarray(SL_RDM.rdm_descriptors["voxel_index"])

    cache = {
        "dissimilarities": np.stack(dissimilarities, axis=0),  # (n_subjects, n_centers, n_pairs) float32
        "voxel_index": voxel_index,
        "mask_shape": mask.shape,
        "n_conditions": n_conditions,
        "method": c.NEURAL_RDM_METHOD,
        "radius": c.RADIUS,
        "threshold": c.THRESHOLD,
    }
    with open(c.SL_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved neural searchlight RDM cache -> {c.SL_CACHE_PATH}")
    print(f"  dissimilarities shape: {cache['dissimilarities'].shape}, dtype: {cache['dissimilarities'].dtype}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
