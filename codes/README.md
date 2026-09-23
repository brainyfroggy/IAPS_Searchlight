# Revised model-RDM searchlight runs — what this folder does and why

This folder reruns the IAPS searchlight RSA (base pipeline: `../searchlight_behavioral_final.ipynb`)
with corrected model RDMs for Valence/Arousal, CLIP-ViT, and GIST. It exists because all
three of the original model RDMs had a specific, identifiable construction problem —
documented below, per stimulus/model — while the neural side of the analysis is
deliberately left untouched. This document is the audit trail: why each change was
made, what was decided against, and where to look if a future pass needs to redo or
extend this.

## What changed, and why

### Valence/Arousal
**Original:** `IAPS_behavioral_RDM/IAPS_behavioral_RDM.ipynb` → `rdm_behavioral.csv` /
`rdm_behavioral_normed.csv`. Built as `pairwise_distances([Mean_Arousal, Mean_Valence], metric='euclidean')`
on the *raw* IAPS norms, then only min-max rescaled afterward (never z-scored).

**Problem:** valence spans nearly the full 1–9 IAPS scale (std ≈ 2.00, range 6.38 across
these 60 stimuli) while arousal is compressed toward the mid-range (std ≈ 1.26, range
4.49). Raw Euclidean distance on two unequally-scaled axes is dominated by the
higher-variance one — so the "joint VA model" was mostly a valence model.

**Verified empirically** (see git history / conversation for the check script): the old
RDM's upper-triangle correlates with a valence-only |Δ| RDM at Spearman ρ ≈ 0.90 vs.
ρ ≈ 0.40 for an arousal-only |Δ| RDM. After the fix, ρ ≈ 0.74 (valence) vs. ρ ≈ 0.62
(arousal) — much more balanced, though valence still contributes somewhat more, which
is expected given IAPS's real valence-arousal correlation structure (the "boomerang"
shape), not an artifact.

**Fix (`run_searchlight_va_zscore.py`):** z-score `Mean_Valence` and `Mean_Arousal`
independently across the 60 stimuli (`scipy.stats.zscore(..., ddof=1)`), *then* Euclidean
distance on the two z-scored columns. Output: `rdm_va_zscore.csv`.

**Explicitly excluded from this step:** separate valence-only / arousal-only searchlights.
These were part of the original methodological discussion (needed for variance
partitioning — see below) but you asked to skip them here; only the joint VA model runs.
Mahalanobis-whitened VA distance (to remove the valence-arousal correlation) was also
considered and rejected per the original discussion — the correlation is a real feature
of the IAPS stimulus set, not sampling noise, so whitening it away would remove signal,
not confound.

### CLIP-ViT
**Original:** `IAPS_Searchlight/searchlight_ai_embeddings_models.ipynb` → `RDM/rdm_clipvit.csv`.
Built as `pairwise_distances(clip_image_embeddigs, metric='correlation')`.

**Problem:** correlation distance mean-centers each embedding vector before comparing —
not the geometry CLIP's contrastive training objective operates under (L2-normalized
embeddings compared by cosine similarity).

**`clip_image_embeddigs.pkl` provenance** (confirmed by reading the extraction cell in
`searchlight_ai_embeddings_models.ipynb`): OpenAI **CLIP ViT-B/32** (`clip.load("ViT-B/32")`),
`model.encode_image(image)` output, shape `(60, 512)`, extracted in the same
`IAPS_60_pnu.csv` stimulus order used throughout this folder. This is genuinely the
CLIP-ViT *image* embedding — distinct from `clip_text_embeddigs.pkl` (CLIP's text
encoder, not used here) and from `pure_vit_embeddings` (a separate plain ImageNet
`ViTModel` from HuggingFace used elsewhere in that same notebook, unrelated to CLIP).
The raw embeddings are **not** pre-normalized (row norms ≈ 9–11, not 1).

**Fix — two variants, run as a deliberate comparison:**
- `run_searchlight_clip_cosine.py`: `pairwise_distances(clip_image_embeddigs, metric='cosine')`
  → `rdm_clip_cosine.csv`. This is CLIP's native geometry.
- `run_searchlight_clip_euclidean_l2norm.py`: L2-normalize each embedding row to unit
  norm, then Euclidean distance → `rdm_clip_euclidean_l2norm.csv`. Not the recommended
  metric on its own — run specifically to check agreement with the cosine version.

**Verified empirically:** the two RDMs' upper-triangle vectors are Spearman ρ = 1.000000
correlated (exactly, to float precision) — the expected result, since squared Euclidean
distance on L2-normalized vectors is a strictly monotonic function of cosine distance
(d² = 2 − 2·cos). Their searchlight t-maps should therefore be effectively identical;
if they visibly diverge, that's a red flag to investigate (e.g. a bug in one script),
not an interesting scientific finding.

### CLIP-ResNet50
**Added August 2026:** `run_searchlight_resnetclip.py` tests the ResNet-backbone CLIP
model used by Wang et al. (2023), implemented as OpenAI CLIP `RN50`. The image encoder
outputs 1024-dimensional image features, matching the model-feature dimensionality
reported for OpenAI CLIP with a ResNet50 backbone in that paper.

The script extracts `RN50` image embeddings from the same 60 IAPS files in
`IAPS_60_pnu.csv` order, saves them to `resnetclip_rn50_image_embeddings.pkl`, then
builds two RDMs:
- `rdm_resnetclip_cosine.csv`: cosine distance on the raw CLIP-RN50 embeddings.
- `rdm_resnetclip_euclidean_l2norm.csv`: Euclidean distance after L2-normalizing each
  embedding row.

The two RDMs are Spearman rho = 1.000000 correlated, so the resulting searchlight maps
are expected to be identical under the rank-based Spearman RSA used here. This was
verified after running: the q<0.01 cosine and L2-normalized Euclidean maps are voxelwise
identical and each contains 35,049 positive voxels after the existing 10-voxel cluster
filter.

### GIST
**Original:** `IAPS_Gist_RDM/searchlight_fmri_gist.ipynb` → `gist_rdm_euc.csv` (also
`gist_rdm.csv`, `gist_rdm_corr_norm.csv`). Raw GIST descriptors (unnormalized oriented
energy, from `generate_gist_rdm.m`) fed straight into Euclidean or correlation distance.

**Problem:** an image that is simply higher-contrast or visually busier produces a
larger-magnitude GIST descriptor regardless of its spatial layout. Unnormalized
Euclidean distance therefore partly indexes overall contrast/energy, not the
spatial-envelope structure GIST is meant to capture.

**Fix (`run_searchlight_gist_l2norm_euclidean.py`):** L2-normalize each image's raw GIST
descriptor to unit norm, then Euclidean distance — the Oliva & Torralba convention,
made into a genuine structural comparison. Descriptors are read from
`IAPS_Gist_RDM/gist/{iaps_id}.jpg.txt` in `IAPS_60_pnu.csv` order (same order the
original notebook already used via `stim_order_txt`). Output: `rdm_gist_l2norm_euclidean.csv`.

## What did NOT change: the neural RDM

`build_neural_searchlight_rdms.py` reproduces `searchlight_behavioral_final.ipynb`'s
neural-RDM step exactly: `get_searchlight_RDMs(subject_data, centers, neighbors, events, method='correlation')`,
radius=5, threshold=0.5. This is the published "acceptable fallback" distance metric for
searchlight RSA (vs. crossnobis), and it is **not being upgraded in this step** — a
decision made explicitly, not an oversight:

- True crossnobis needs multivariate noise normalization (MVNN) from GLM residual
  covariance. Only `ResMS.img` (a single residual-*variance* summary per voxel, no
  cross-voxel covariance) exists per subject in `IAPS_fMRI_RSA/GLM_betas/GLM1-20` — no
  full residual timeseries (`Res_*.img`) were saved by the original SPM first-level GLMs.
- `allsub_avg.npy` (the data these scripts load) already collapses across all 5 runs
  per condition before this point, so run identity needed for cross-validated distance
  is also gone.
- Doing this properly means re-running first-level GLMs in SPM/MATLAB with residual
  volumes saved, then rebuilding the averaging step to preserve per-run patterns — a
  separate, larger piece of work, deferred.

Second-level metric is unchanged too: Spearman correlation between each searchlight's
neural RDM and the model RDM (`evaluate_models_searchlight(..., method='spearman')`),
matching the existing convention throughout this project.

## What is explicitly out of scope in the four single-model scripts
- The older variance-partitioning notebooks (`partial_rsa_variance_partitioning*.ipynb`,
  `searchlight_partial_variance*.ipynb`) remain untouched and still use their old RDMs.
  A corrected whole-brain multiple-regression RSA has now been added separately in
  `run_searchlight_multiple_regression.py`; see the dedicated section below. It estimates
  unique standardized beta coefficients but does not perform a full commonality/shared-
  variance decomposition.
- No noise ceiling.
- No valence-only / arousal-only searchlight (joint VA model only).
- Crossnobis/MVNN (see above).

These were all considered as part of the original methodological discussion but scoped
out for this run — a future pass can pick them back up using the corrected RDMs
produced here.

## Whole-brain multiple-regression RSA: CLIP + GIST + VA

`run_searchlight_multiple_regression.py` was added to test whether the very broad CLIP
map is explained only by structure shared with low-level GIST features or valence/arousal.
It runs two three-predictor models at every searchlight and for every subject:

1. neural RDM ~ CLIP cosine + GIST L2-normalized Euclidean + **raw VA Euclidean**
2. neural RDM ~ CLIP cosine + GIST L2-normalized Euclidean + **z-scored VA Euclidean**

The raw VA RDM rebuilt by this script was verified to be exactly equal to the historical
`RDM/rdm_behavioral.csv`. CLIP Euclidean-on-L2-normalized embeddings was not entered as
a second predictor because its ranks are identical to CLIP cosine distance; including
both would make the design redundant.

All three predictor RDM vectors and each neural RDM vector are rank transformed and
standardized. The resulting coefficients are therefore standardized multiple-regression
betas in a Spearman-RSA model. A positive CLIP beta means that CLIP explains neural-RDM
structure after controlling simultaneously for GIST and VA. The implementation uses the
identity `beta = inverse(Rxx) * rxy`, where `rxy` is the set of already-computed
single-model Spearman correlations. Direct rank-OLS checks at representative subjects and
searchlights agreed to numerical precision (maximum absolute error < 3e-16).

Predictor collinearity was low:

| VA version | rho(CLIP, VA) | rho(CLIP, GIST) | largest VIF | condition number |
|---|---:|---:|---:|---:|
| raw VA | 0.2910 | -0.0177 | 1.094 | 1.83 |
| z-scored VA | 0.2817 | -0.0177 | 1.087 | 1.79 |

### Multiple-regression group inference

The primary correction is a one-sided voxelwise max-T FWE test across all 56,949
searchlight centers. It uses 10,000 synchronized subject sign-flip permutations: within
each permutation, the same subject signs are applied to every voxel, preserving the
observed spatial dependence between overlapping searchlights. Maps are thresholded at
`p_FWE < 0.05`; no additional arbitrary cluster extent is applied to these primary maps.

For continuity with previous results, secondary maps are also saved using two-sided
one-sample t-test p-values, BH-FDR `q < 0.01` or `q < 0.001`, positive coefficients only,
and a 30-voxel face-connected cluster extent filter. These FDR maps are secondary; the
fixed 30-voxel filter is not itself a cluster-level FWE correction.

Primary max-T FWE results:

| VA covariate | unique CLIP voxels | unique GIST voxels | unique VA voxels |
|---|---:|---:|---:|
| raw VA | 15,093 | 4,289 | 0 |
| z-scored VA | 14,854 | 4,330 | 46 |

The two unique-CLIP maps are highly stable across the VA definition: 14,653 voxels
overlap (Dice = 0.9786). The z-scored-VA unique-CLIP max-T map contains 14,854 voxels,
compared with 34,110 voxels in the earlier single-model CLIP BH-FDR `q < 0.01` map — a
56.5% reduction. The remaining CLIP effect is still broad, so shared variance with GIST
and VA is not sufficient to explain the cortex-wide result. Overlapping radius-5
searchlights also make all searchlight maps spatially extended.

The complete numerical audit is in `multiple_regression_summary.json`; subject beta
arrays, full-model R-squared arrays, raw/group p-values, max-T null distributions, and
unthresholded t-statistics are kept in `codes/` so thresholds can be audited without
rerunning the RSA.

## Design: two-stage pipeline (why a shared cache script)

`get_searchlight_RDMs(..., method='correlation')` depends only on the fMRI data and
mask — not on which model RDM it's later compared against. Since all 4 model
comparisons need the exact same neural computation (~1–3 hours across 20 subjects),
recomputing it 4 times would be pure waste, and running 4 copies of
`evaluate_models_searchlight`'s internal `n_jobs=6` simultaneously risks oversubscribing
the machine's cores. So:

- **Stage 1 (`build_neural_searchlight_rdms.py`, run once):** computes the per-subject
  searchlight RDMs and caches them to `SL_RDM_allsubjects.pkl` — a dict with keys
  `dissimilarities` (`(n_subjects, n_centers, n_pairs)` `float32` array — float32 to keep
  the cache to a manageable size), `voxel_index`, `mask_shape`, `n_conditions`, `method`,
  `radius`, `threshold`.
- **Stage 2 (`run_searchlight_*.py`, run in parallel):** each script loads that cache,
  re-hydrates it into an `rsatoolbox.rdm.RDMs` object per subject (so the actual
  Spearman-vs-RDM comparison reuses rsatoolbox's own tested code via
  `evaluate_models_searchlight`/`eval_fixed`, rather than a hand-rolled reimplementation),
  builds its own model RDM, and only needs to run the comparison + group stats — minutes,
  not hours.

**Do not run any `run_searchlight_*.py` script before `build_neural_searchlight_rdms.py`
has completed** — they will raise `FileNotFoundError` on `SL_RDM_allsubjects.pkl` if run first.

`common.py` holds the shared paths/constants and the group-level statistics routine
(`group_stats_and_save`) so that logic exists in exactly one place across all 4 model
scripts, rather than being copy-pasted and risking drift.

## Statistics: FDR levels and cluster threshold

Per your instruction, the same statistical recipe as the base notebook — Fisher-z
transform of per-subject Spearman correlations, one-sample t-test across subjects
(`scipy.stats.ttest_1samp`), positive tail only (t < 0 set to NaN, matching existing
convention throughout this project) — is computed once per model, then FDR-corrected
independently at **three alpha levels: 0.05, 0.01, 0.001** (`statsmodels.stats.multitest.fdrcorrection`,
`method='indep'`), each followed by a **10-voxel cluster threshold**
(`nilearn.image.threshold_img(..., cluster_threshold=10)`) — same cluster size used
elsewhere in this project. All three constants (`FDR_ALPHAS`, `CLUSTER_THRESHOLD`) live
in `common.py` if a different threshold/cluster size is needed later.

Unthresholded intermediates are saved next to the scripts (`tstat_{model}.npy`,
`pvalue_{model}.npy`, `eval_score_{model}.csv`) so re-thresholding at a different alpha
or cluster size doesn't require rerunning the searchlight.

### August 2026 note: old `beh60` ROI vs corrected z-scored VA map

Two Valence/Arousal-related maps now exist in `../outputs/tBrainmap/final_results2/`,
and they should not be treated as interchangeable:

- `tmap_va_zscore_euclidean_fdr01_cluster30.nii.gz`: the corrected joint VA model from
  this folder. Valence and arousal are independently z-scored across the 60 stimuli,
  Euclidean distance is computed on those z-scored axes, then the map is thresholded at
  BH-FDR `q < 0.01`, positive t-values only, with a 30-voxel cluster threshold.
- `tmap_beh60_raw_euclidean_fdr01_cluster30_motor_removed.nii.gz`: an exact copy of the
  older `outputs/tBrainmap/final_results/tmap_beh60_p05_v2.nii.gz`, saved into
  `final_results2/` with a clearer name. This older ROI comes from the raw
  Valence/Arousal Euclidean behavioral RDM (`rdm_behavioral_normed.csv`) and includes
  the previous manual removal of motor-region voxels. Despite the historical `p05_v2`
  name, provenance checks showed it is a positive-tail BH-FDR `q < 0.01` map with a
  30-voxel cluster threshold, followed by manual motor-ROI removal.
- `tmap_beh60_raw_euclidean_fdr001_cluster30_motor_removed.nii.gz`: the stricter
  raw-VA companion map made from the same saved old behavioral searchlight score table
  (`outputs/searchlight_results/20sub_SL_eval_score_norm_v3.csv`) and voxel-center
  indices (`outputs/voxel_center_id_beh60_61997.npy`). The recipe is Fisher z,
  one-sample t-test across 20 subjects, BH-FDR `q < 0.001`, positive t-values only,
  30-voxel cluster threshold, then intersection with the motor-removed raw-VA mask.
  It contains one 53-voxel cluster; all 53 voxels are inside the `q < 0.01`
  motor-removed map.
- `tmap_beh60_raw_euclidean_fdr005_cluster30_motor_removed.nii.gz`: same recipe as the
  `q < 0.001` raw-VA motor-removed map, but thresholded at BH-FDR `q < 0.005`. It
  contains 298 voxels in two clusters and is nested between the `q < 0.001` and
  `q < 0.01` motor-removed maps.
- `tmap_beh60_raw_euclidean_fdr01_cluster30_motor_included.nii.gz`: an exact copy of
  `outputs/tBrainmap/tmap_beh60_p05_v3_fisherz.nii.gz`, saved as the same raw-VA
  positive-tail ROI before the manual motor-region exclusion. It contains 1082 positive
  voxels: the 1038 voxels in the motor-removed map plus 44 voxels that were excluded
  manually in `p05_v2`.
- `tmap_clip_cosine_fdr01_top1038voxels.nii.gz`,
  `tmap_clip_euclidean_l2norm_fdr01_top1038voxels.nii.gz`, and
  `tmap_gist_l2norm_euclidean_fdr01_top1038voxels.nii.gz`: equal-size comparison maps
  made by taking the 1038 largest positive t-values from each model's existing `fdr01`
  map, where 1038 is the voxel count of
  `tmap_beh60_raw_euclidean_fdr01_cluster30_motor_removed.nii.gz`. These are top-N
  maps, not new FDR thresholds. The two CLIP top-N maps are voxelwise identical.

These two maps look different because the model RDM is different. The old raw Euclidean
behavioral RDM is valence-dominated: valence has a larger spread than arousal in this
60-image IAPS set, so raw Euclidean distance gives valence more weight. The corrected
`va_zscore` RDM balances the two dimensions before computing Euclidean distance, so
arousal contributes more to the model geometry. A voxel-level check confirmed the
difference: old reproduced map = 1038 voxels, corrected z-scored VA cluster-30 map =
1241 voxels, overlap = 736 voxels, old-only = 302 voxels, z-scored-only = 505 voxels.

Use the corrected `va_zscore` map for the revised RSA analysis. Use the reproduced
`beh60_raw_euclidean...motor_removed` map only when the goal is to match or compare
against the historical ROI.

## Output provenance

| Output (in `../outputs/tBrainmap/final_results2/`) | Produced by | Input RDM CSV |
|---|---|---|
| `tmap_va_zscore_euclidean_fdr{05,01,001}.nii.gz` | `run_searchlight_va_zscore.py` | `rdm_va_zscore.csv` |
| `tmap_va_zscore_euclidean_fdr01_cluster30.nii.gz` | `run_searchlight_va_zscore.py` / re-thresholded existing VA `fdr01` t-map | `rdm_va_zscore.csv` |
| `tmap_beh60_raw_euclidean_fdr01_cluster30_motor_removed.nii.gz` | exact renamed copy of historical `final_results/tmap_beh60_p05_v2.nii.gz` | historical `rdm_behavioral_normed.csv` |
| `tmap_beh60_raw_euclidean_fdr001_cluster30_motor_removed.nii.gz` | recomputed from historical behavioral subject-score table, then intersected with motor-removed mask | historical `rdm_behavioral_normed.csv` |
| `tmap_beh60_raw_euclidean_fdr005_cluster30_motor_removed.nii.gz` | recomputed from historical behavioral subject-score table, then intersected with motor-removed mask | historical `rdm_behavioral_normed.csv` |
| `tmap_beh60_raw_euclidean_fdr01_cluster30_motor_included.nii.gz` | exact renamed copy of historical `outputs/tBrainmap/tmap_beh60_p05_v3_fisherz.nii.gz` | historical `rdm_behavioral_normed.csv` |
| `tmap_clip_cosine_fdr01_top1038voxels.nii.gz` | top 1038 positive t-values from `tmap_clip_cosine_fdr01.nii.gz` | `rdm_clip_cosine.csv` |
| `tmap_clip_euclidean_l2norm_fdr01_top1038voxels.nii.gz` | top 1038 positive t-values from `tmap_clip_euclidean_l2norm_fdr01.nii.gz` | `rdm_clip_euclidean_l2norm.csv` |
| `tmap_gist_l2norm_euclidean_fdr01_top1038voxels.nii.gz` | top 1038 positive t-values from `tmap_gist_l2norm_euclidean_fdr01.nii.gz` | `rdm_gist_l2norm_euclidean.csv` |
| `tmap_clip_cosine_fdr{05,01,001}.nii.gz` | `run_searchlight_clip_cosine.py` | `rdm_clip_cosine.csv` |
| `tmap_clip_euclidean_l2norm_fdr{05,01,001}.nii.gz` | `run_searchlight_clip_euclidean_l2norm.py` | `rdm_clip_euclidean_l2norm.csv` |
| `tmap_gist_l2norm_euclidean_fdr{05,01,001}.nii.gz` | `run_searchlight_gist_l2norm_euclidean.py` | `rdm_gist_l2norm_euclidean.csv` |
| `tmap_resnetclip_cosine_fdr01.nii.gz` | `run_searchlight_resnetclip.py` | `rdm_resnetclip_cosine.csv` |
| `tmap_resnetclip_euclidean_l2norm_fdr01.nii.gz` | `run_searchlight_resnetclip.py` | `rdm_resnetclip_euclidean_l2norm.csv` |
| `tmap_mrsa_clip_gist_va_{raw,zscore}_beta_{clip,gist,va}_maxT_fwe05.nii.gz` | `run_searchlight_multiple_regression.py` | CLIP cosine + GIST L2-Euclidean + raw/z-scored VA |
| `tmap_mrsa_clip_gist_va_{raw,zscore}_beta_{clip,gist,va}_fdr{01,001}_cluster30.nii.gz` | `run_searchlight_multiple_regression.py` (secondary maps) | CLIP cosine + GIST L2-Euclidean + raw/z-scored VA |
| `mean_r2_mrsa_clip_gist_va_{raw,zscore}.nii.gz` | `run_searchlight_multiple_regression.py` (descriptive; not thresholded inference) | CLIP cosine + GIST L2-Euclidean + raw/z-scored VA |

All RDM CSVs and `SL_RDM_allsubjects.pkl` are written into this folder (`codes/`), not
`../outputs/`, to keep this run's intermediates self-contained and separate from the
original pipeline's outputs.

## How to run

```
# Stage 1 — once, ~1-3 hours
python build_neural_searchlight_rdms.py

# Stage 2 — after Stage 1 completes, these 4 can run in parallel (separate processes/terminals)
python run_searchlight_va_zscore.py
python run_searchlight_clip_cosine.py
python run_searchlight_clip_euclidean_l2norm.py
python run_searchlight_gist_l2norm_euclidean.py
python run_searchlight_resnetclip.py

# After Stage 1 and the corrected single-model score tables exist:
python run_searchlight_multiple_regression.py
```
