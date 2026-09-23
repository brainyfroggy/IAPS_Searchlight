# IAPS Searchlight RSA

## Purpose

This repository holds the analysis code for a whole-brain searchlight
representational-similarity analysis (RSA) of fMRI responses to the
International Affective Picture System (IAPS) emotional-image set (60 stimuli,
20 subjects). At every searchlight center, a neural representational
dissimilarity matrix (RDM) built from BOLD response patterns is compared
against model RDMs derived from image-computable models (CLIP, GIST,
VGG16/19, ViT, DINO, DeepGaze II, EmoNet) and from human/AI-generated
valence-arousal (VA) ratings of the stimuli, using Spearman-correlation RSA
(`rsatoolbox`). Whole-brain multiple-regression RSA and variance-partitioning
analyses test whether a broad model effect (e.g. CLIP) is explained by shared
structure with a lower-level (GIST) or affective (VA) predictor. Group-level
results are FDR- or max-T-FWE-corrected statistical maps, rendered onto the
cortical surface with `pycortex` and visualized with interactive Plotly/HTML
brain viewers.

This is a research-code-only upload: no raw IAPS images, fMRI data, or
computed intermediates (RDMs, feature caches, statistical maps) are included.
It exists so the analysis pipeline itself is inspectable and reproducible
against a matching data layout, not so it can be re-run out of the box.

**Note on file cleanup:** a prior pass removed several files that did not
represent finished, working analyses — a completely empty notebook
(`nilearn_second_level.ipynb`), a corrupted stub (`example_Cichy2016.ipynb`,
which contained only a bare list of `import` statements and was not valid
notebook JSON — apparently a fragment saved with the wrong extension rather
than a working tutorial adaptation of Cichy et al. 2016), an ad hoc
reshape/orientation scratch notebook (`tmp/brain_plot.ipynb`), a grab-bag of
unrelated one-off snippets (`misc.ipynb`), an empty RDM-correlation notebook
(`RDM/RDM_correlation.ipynb`), an archived DeepGaze searchlight attempt that
hit an unresolved `torch.fx` tracing error and was superseded by
`searchlight_DGii.ipynb` (`archive/searchlight_dg2.ipynb`), two searchlight
decoding notebooks that trailed off mid-computation with unresolved errors
(`SL_decoding/plot.ipynb`, `SL_decoding/sl_decoding.ipynb` — superseded by the
complete standalone script `SL_decoding/searchlightpt3.py`), and a duplicate
pycortex notebook with its outputs stripped
(`visualization/pycortex/surface_plot_all.ipynb`, identical in source to
`surface_plot_all_executed.ipynb`). Only complete, working files remain.

## Contents

### RSA searchlight analyses, one per feature model (root directory)

The bulk of the repository is a set of parallel notebooks that apply the same
searchlight-RSA method to different candidate models of the neural
representational geometry. Each one loads the cached neural searchlight RDMs,
builds a model RDM from a different feature space, correlates them at every
searchlight center, and produces group-level statistical maps:

- **CNN / vision-model features:** `searchlight_vgg16.ipynb`,
  `searchlight_vgg16_3emo.ipynb`, `searchlight_vgg16_content6cate.ipynb`,
  `searchlight_vgg19.ipynb` (VGG layer activations), `searchlight_DINO_v1v2.ipynb`,
  `searchlight_DINO_v3.ipynb` (DINO self-supervised ViT features),
  `searchlight_DGii.ipynb` (DeepGaze II saliency-model features),
  `searchlight_gist.ipynb`, `searchlight_gist_categorical.ipynb` (GIST
  low-level scene descriptors).
- **CLIP / language-vision and AI-description models:**
  `searchlight_ai_embeddings_models.ipynb` (CLIP text/image, ViT embeddings),
  `searchlight_ai_multimodels.ipynb`, `searchlight_ai_va.ipynb`,
  `searchlight_ai_descriptions_clip.ipynb`,
  `searchlight_ai_descriptions_clip_permutation.ipynb` (permutation-test
  variant), `searchlight_ai_descriptions_VA_removed.ipynb` (CLIP model with
  valence-arousal variance regressed out).
- **Emotion model:** `searchlight_emonet.ipynb` (EmoNet features).

### Behavioral / affective-dimension searchlight RSA (root directory)

Notebooks comparing the neural RDM against human behavioral ratings instead of
computed image features — different affective dimensions and stimulus
subsets, each following the same searchlight-then-group-stats template:
`searchlight_behavioral.ipynb`, `searchlight_behavioral_norm.ipynb`,
`searchlight_behavioral_final.ipynb`, `searchlight_behavioral_final2.ipynb`,
`searchlight_behavioral_final2_fdr05.ipynb`,
`searchlight_behavioral_content_category.ipynb`,
`searchlight_behavioral_emo_category.ipynb`,
`searchlight_behavioral_pleasant_unpleasant_separate_fdr01.ipynb`,
`searchlight_behavioral_pleasant_unpleasant_separate_fdr05.ipynb`,
`searchlight_arousal.ipynb`, `searchlight_valence.ipynb`,
`searchlight_valence_arousal_separate.ipynb`, `searchlight_pleasure.ipynb`,
`searchlight_unpleasure.ipynb`, `searchlight_pleasant_unpleasant.ipynb`,
`searchlight_neutral.ipynb`, `searchlight_stim(erotic).ipynb`,
`searchlight_appraisal.ipynb`, and `searchlight_glm.ipynb` (first/second-level
GLM prep feeding the behavioral analyses).

A separate radius-sensitivity sweep repeats the behavioral searchlight at two
radius ranges: `searchlight_behavioral_radius_1_to_5.ipynb` /
`run_searchlight_behavioral_radius_1_to_5.py` and
`searchlight_behavioral_radius_6_to_10.ipynb` /
`run_searchlight_behavioral_radius_6_to_10.py`, summarized by
`plot_radius_1_to_10_fdr_voxel_counts.py`.

### Variance partitioning and multi-model comparison (root directory)

Notebooks that test whether one model's searchlight effect is explained by
shared variance with another model, rather than treating each model in
isolation: `partial_rsa_variance_partitioning.ipynb`,
`partial_rsa_variance_partitioning_VA_CLIP_t95.ipynb`,
`searchlight_partial_variance.ipynb`,
`searchlight_partial_variance_appraisal_multimodel.ipynb`,
`searchlight_partial_variance_emonet_multimodel.ipynb`,
`searchlight_partial_variance_va_multimodel.ipynb`,
`comparison_ai_human_va.ipynb` (AI-derived vs. human VA ratings), and
`compare_brainmaps.ipynb` / `searchlight_roi_overlap.ipynb` (overlap between
thresholded statistical maps from different models).

### `codes/` — revised, audited pipeline

A later, corrected re-run of the core searchlight RSA: fixes the construction
of the VA, CLIP, and GIST model RDMs (documented problem-by-problem), adds a
whole-brain multiple-regression RSA and a CLIP-ResNet50 comparison, and
splits the pipeline into a one-time neural-RDM caching stage
(`build_neural_searchlight_rdms.py`) plus per-model comparison scripts
(`run_searchlight_*.py`) that share constants/statistics code (`common.py`).
See `codes/README.md` for the full rationale, statistics, and run order —
it is a detailed audit trail of what changed and why.

### `SL_decoding/` — searchlight decoding

`searchlightpt3.py` is a complete, standalone script implementing a
decoding-based (SVM classification accuracy per searchlight sphere) analysis,
complementary to the RSA-based approach used everywhere else in this
repository: it loads per-subject beta maps, builds searchlight spheres,
runs parallelized cross-validated SVM decoding at each sphere, and saves a
results array plus a summary heatmap.

### `visualization/` — figures and brain-surface rendering

- `figures_manscript.ipynb` — manuscript figure generation.
- `ROI_visualisations.ipynb` / `.py`, `ROI_visualisations_simona.ipynb` — ROI
  overlay visualizations.
- `plot_positive_axial_tmaps.py` — axial-slice t-map plotting.
- `interactive_brain_controls.html`, `interactive_brain_controls_clean_legend.html`,
  `plotly_3D_volumetric_ROIs_GIST_Scene_Emotion.html` (and the `_AAL3labels`
  and `_interactive` variants) — standalone, self-contained Plotly/HTML brain
  viewers; open any of them directly in a browser, no server or dependencies
  needed.
- `pycortex/` — cortical surface-rendering scripts and notebooks:
  `surface_plot.py` is a minimal quickstart example of the pycortex workflow;
  `surface_plot.ipynb` is the full flatmap-rendering analysis;
  `surface_plot_all_executed.ipynb` is that pipeline applied across all
  models/maps (kept with its run outputs, so this is the version to view);
  `surface_plot_oxford_harvard.ipynb` / `.py` and
  `harvard_oxford_atlas_overlay.py` render Harvard-Oxford-atlas-labeled
  flatmaps for the GIST, CLIP-ViT, and VA statistical maps.

## How to Use

This repository does not include stimulus images, fMRI data, or any computed
intermediate (RDMs, feature caches, statistical maps). To rerun any part of
the pipeline you will need your own copy of:

- The IAPS stimulus set and single-trial/beta fMRI volumes (per-subject
  `.mat`/NIfTI beta images), laid out to match the hard-coded paths near the
  top of each notebook/script (these currently point at the original lab file
  server — update them for your environment).
- A stimulus-order CSV (`IAPS_60_pnu.csv` or similar) giving the mapping
  between stimulus IDs and image/behavioral-rating order.
- Pretrained weights/checkpoints for whichever feature model(s) you want to
  extract (CLIP, VGG16/19, DINO, DeepGaze II, EmoNet) — these are loaded
  from each model's own public release, not vendored here.

General order of operations:

1. Start with `codes/` if you want the current, corrected pipeline — run
   `build_neural_searchlight_rdms.py` once, then the `run_searchlight_*.py`
   scripts (see `codes/README.md` for exact commands and required run order).
2. The root-level `searchlight_*.ipynb` notebooks are self-contained per
   model/behavioral-dimension: each one loads or recomputes the neural RDMs,
   builds its model RDM, runs the searchlight comparison, and performs
   group-level statistics (Fisher-z, one-sample t-test, FDR correction) —
   run cells top to bottom.
3. Use `visualization/` notebooks/scripts afterward, pointed at your own
   thresholded statistical maps, to render brain-surface figures or open the
   `.html` files directly for interactive 3D viewers (no data needed for the
   HTML viewers themselves — they embed a previously rendered figure).

Every notebook/script writes its outputs (RDM `.csv`, `.pkl` feature caches,
`.npy` arrays, statistical map `.nii.gz` files) next to itself or into an
`outputs/` folder referenced in the code — none of these are included, so
expect to regenerate them.

## Dependencies

- Standard scientific Python stack: `numpy`, `scipy`, `pandas`, `statsmodels`,
  `scikit-learn`, `nibabel`, `nilearn`, `rsatoolbox`.
- [`pycortex`](https://github.com/gallantlab/pycortex) for cortical surface
  visualization (`visualization/pycortex/`).
- `torch`/`torchvision`, `tqdm`, `seaborn`, `matplotlib`, `Pillow` for feature
  extraction and plotting.
- **DeepGaze** — `searchlight_DGii.ipynb` depends on
  [matthias-k/DeepGaze](https://github.com/matthias-k/DeepGaze), which is not
  vendored in this repository. Install it separately per its own
  instructions if you need to rerun that notebook.
- Model embeddings (CLIP, GIST, VGG, ViT, DINO, ResNet, EmoNet) are extracted
  from the IAPS stimulus images using each model's own published
  weights/checkpoints; see the individual notebooks for the extraction cells.

## Caveats / not included

This repository contains code and documentation only. The following are
intentionally excluded and are **not** in version control here:
- Raw IAPS stimulus images and any derived image data — the IAPS image set is
  copyright-restricted and cannot be redistributed.
- All computed intermediates and results: neural/model RDM `.csv` files,
  `.pkl` feature caches, `.npy` arrays, log files, JSON summaries, and result
  images/figures (`.png`/`.svg`, other than the ones embedded in the
  standalone `.html` viewers).
- Third-party model repositories (e.g. a full clone of DeepGaze) — install
  these separately per Dependencies above.

Anyone re-running this pipeline will need to regenerate the neural/model RDMs
and feature caches from their own copy of the IAPS stimulus set and fMRI
data; file paths in some notebooks are hard-coded to the original lab file
server and will need updating.
