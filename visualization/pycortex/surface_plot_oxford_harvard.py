"""Render the three individual t-value flatmaps with shared ROI overlays.

The statistical maps are projected onto fsaverage and each receives the same
selected AAL3, Wang--Kastner, and HCP--MMP labels as the categorical Figure 3C
functional-ROI overlap flatmap.
"""

from pathlib import Path
import re
import sys

import cortex
import cortex.database
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from nilearn import datasets, surface
from PIL import Image
from scipy import sparse


# -----------------------------------------------------------------------------
# Paths and display settings
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(
    "/mnt/n/Experimental_Data/yujunchen/projects/IAPS_Searchlight"
)
PY_CORTEX_DIR = PROJECT_DIR / "visualization/pycortex"
MANUSCRIPT_FIGURES_DIR = Path(
    "/mnt/c/Users/yujunchen/OneDrive - University of Florida/"
    "BME departmental/manuscript/Emotion Semantic Categorization/"
    "manuscript_figures"
)

sys.path.insert(0, str(PY_CORTEX_DIR))
from harvard_oxford_atlas_overlay import (
    add_requested_aparc_overlay,
    add_requested_aal3_overlay,
    add_requested_hcp_mmp_overlay,
    add_requested_wang2015_overlay,
)

CLIP_TMAP = (
    PROJECT_DIR
    / "outputs/tBrainmap/final_results3/tmap_clip_euclidean_l2norm_fdr01.nii.gz"
)
GIST_TMAP = (
    PROJECT_DIR
    / "outputs/tBrainmap/final_results3/tmap_gist_l2norm_euclidean_fdr01_visual_largest_component_zero_bg.nii.gz"
)
VA_TMAP = (
    PROJECT_DIR
    / "outputs/tBrainmap/final_results/tmap_beh60_p05_v2.nii.gz"
)

OUT_CLIP = PY_CORTEX_DIR / "surface_plot_oxford_harvard_clipvit.png"
OUT_GIST = PY_CORTEX_DIR / "surface_plot_oxford_harvard_gist.png"
OUT_VA = PY_CORTEX_DIR / "surface_plot_oxford_harvard_valence_arousal.png"
OUT_COMBINED = PY_CORTEX_DIR / "surface_plot_oxford_harvard_three_maps.png"
OUT_LATEST = PY_CORTEX_DIR / "surface_plot_oxford_harvard.png"
OUT_SUMMARY = PY_CORTEX_DIR / "surface_plot_oxford_harvard_roi_summary.csv"
OUT_MANUSCRIPT_COMBINED = (
    MANUSCRIPT_FIGURES_DIR / "Figure_3C_final_results3_flatmaps.png"
)
OUT_MANUSCRIPT_LATEST = (
    MANUSCRIPT_FIGURES_DIR / "Figure_3C_final_results3_flatmaps_latest.png"
)

ATLAS_NAME = "cort-maxprob-thr25-1mm"
TMAP_THRESHOLD = 0.01
DISPLAY_VMAX_PERCENTILE = 98
VA_DISPLAY_VMAX = 7.0
MIN_ACTIVE_VERTICES = 25
MIN_ACTIVE_PERCENT = 0.50
SMOOTHING_STEPS = 3
SMOOTHING_CUTOFF = 0.48
# Full flatmaps are retained in the revised vertical manuscript panel.

ROI_COLORS = [
    "#E76F51", "#F4A261", "#E9C46A", "#A7C957", "#2A9D8F",
    "#22A7B8", "#3A86C8", "#6C8CD5", "#8E7DBE", "#B07AA1",
    "#D17A9B", "#F08A8B", "#78A6A3", "#C2A878",
    "#7F8C8D", "#D4A373", "#90BE6D", "#577590", "#B56576",
    "#6D597A", "#43AA8B", "#F8961E", "#277DA1", "#BC6C25",
]

# Panel-specific anatomical references requested for the final three-map
# figure.  The Scene panel retains the complete shared atlas layer; Emotion
# and GIST show only the parcels relevant to their respective results.
EMOTION_OVERLAY_GROUPS = {
    "aparc": (
        ("STS", ("bankssts",)),
        ("SMG", ("supramarginal",)),
    ),
    "hcp_mmp": (
        ("RSC", ("RSC",)),
    ),
    "wang2015": (
        ("LO", (14, 15)),
        ("MT", (12, 13)),
    ),
}
GIST_OVERLAY_GROUPS = {
    "wang2015": (
        ("V1", (1, 2)),
        ("V2", (3, 4)),
        ("V3", (5, 6)),
        ("V4", (7,)),
        ("VO", (8, 9)),
        ("LO", (14, 15)),
        ("MT", (12, 13)),
    ),
}

# Keep the Harvard-Oxford outline/label layer visually matched to the
# manuscript's original matched-1038 Figure 3C panels.  The final-results3 CLIP
# and GIST maps are broader than the matched-1038 maps; if we recompute the
# displayed parcels from those broad maps, the VA panel is flooded with unrelated
# labels/outlines even though the VA t-map is the same.  This fixed list is the
# displayed parcel union from Figure_3C_matched1038_roi_summary.csv.
DISPLAY_REGIONS = [
    "Lateral Occipital Cortex, inferior division",
    "Lateral Occipital Cortex, superior division",
    "Occipital Pole",
    "Middle Temporal Gyrus, temporooccipital part",
    "Occipital Fusiform Gyrus",
    "Temporal Occipital Fusiform Cortex",
    "Angular Gyrus",
    "Inferior Temporal Gyrus, temporooccipital part",
    "Superior Parietal Lobule",
    "Supramarginal Gyrus, posterior division",
    "Lingual Gyrus",
]


SHORT_LABELS = {
    "Frontal Pole": "FP",
    "Insular Cortex": "Insula",
    "Superior Frontal Gyrus": "SFG",
    "Middle Frontal Gyrus": "MFG",
    "Inferior Frontal Gyrus, pars triangularis": "IFGtri",
    "Inferior Frontal Gyrus, pars opercularis": "IFGop",
    "Precentral Gyrus": "Precentral",
    "Temporal Pole": "TP",
    "Superior Temporal Gyrus, anterior division": "aSTG",
    "Superior Temporal Gyrus, posterior division": "pSTG",
    "Middle Temporal Gyrus, anterior division": "aMTG",
    "Middle Temporal Gyrus, posterior division": "pMTG",
    "Middle Temporal Gyrus, temporooccipital part": "toMTG",
    "Inferior Temporal Gyrus, anterior division": "aITG",
    "Inferior Temporal Gyrus, posterior division": "pITG",
    "Inferior Temporal Gyrus, temporooccipital part": "toITG",
    "Postcentral Gyrus": "Postcentral",
    "Superior Parietal Lobule": "SPL",
    "Supramarginal Gyrus, anterior division": "aSMG",
    "Supramarginal Gyrus, posterior division": "pSMG",
    "Angular Gyrus": "AG",
    "Lateral Occipital Cortex, superior division": "sLOC",
    "Lateral Occipital Cortex, inferior division": "iLOC",
    "Intracalcarine Cortex": "ICC",
    "Frontal Medial Cortex": "FMC",
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": "SMA",
    "Subcallosal Cortex": "Subcallosal",
    "Paracingulate Gyrus": "ParaCG",
    "Cingulate Gyrus, anterior division": "aCG",
    "Cingulate Gyrus, posterior division": "pCG",
    "Precuneous Cortex": "Precuneus",
    "Cuneal Cortex": "Cuneus",
    "Frontal Orbital Cortex": "OFC",
    "Parahippocampal Gyrus, anterior division": "aPHG",
    "Parahippocampal Gyrus, posterior division": "pPHG",
    "Lingual Gyrus": "LG",
    "Temporal Fusiform Cortex, anterior division": "aTFG",
    "Temporal Fusiform Cortex, posterior division": "pTFG",
    "Temporal Occipital Fusiform Cortex": "toFG",
    "Occipital Fusiform Gyrus": "oFG",
    "Frontal Operculum Cortex": "FrOP",
    "Central Opercular Cortex": "COp",
    "Parietal Operculum Cortex": "POp",
    "Planum Polare": "PP",
    "Heschl's Gyrus (includes H1 and H2)": "HG",
    "Planum Temporale": "PT",
    "Supracalcarine Cortex": "SCC",
    "Occipital Pole": "OP",
}


def base_region(label):
    """Remove the hemisphere prefix used by symmetric-split atlas labels."""
    return re.sub(r"^(Left|Right)\s+", "", label).strip()


def load_tmap_texture(path, fsaverage):
    """Project a volumetric t-map through the cortical ribbon."""
    left = surface.vol_to_surf(
        path,
        fsaverage.pial_left,
        inner_mesh=fsaverage.white_left,
        interpolation="linear",
        n_samples=7,
    )
    right = surface.vol_to_surf(
        path,
        fsaverage.pial_right,
        inner_mesh=fsaverage.white_right,
        interpolation="linear",
        n_samples=7,
    )
    texture = np.hstack([left, right])
    texture[np.abs(texture) < TMAP_THRESHOLD] = np.nan
    return texture


def project_atlas(atlas_img, fsaverage):
    """Project discrete atlas labels using modal sampling through the ribbon."""
    left = surface.vol_to_surf(
        atlas_img,
        fsaverage.pial_left,
        inner_mesh=fsaverage.white_left,
        interpolation="nearest_most_frequent",
        n_samples=9,
    )
    right = surface.vol_to_surf(
        atlas_img,
        fsaverage.pial_right,
        inner_mesh=fsaverage.white_right,
        interpolation="nearest_most_frequent",
        n_samples=9,
    )
    return np.nan_to_num(np.hstack([left, right]), nan=0).astype(np.int16)


def normalized_mesh_adjacency(n_vertices, triangles):
    """Sparse row-normalized adjacency used for topology-aware mask cleanup."""
    edges = np.vstack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    edges = np.vstack([edges, edges[:, ::-1]])
    rows = np.hstack([edges[:, 0], np.arange(n_vertices)])
    cols = np.hstack([edges[:, 1], np.arange(n_vertices)])
    weights = np.hstack([np.ones(len(edges)), np.full(n_vertices, 2.0)])
    graph = sparse.coo_matrix(
        (weights, (rows, cols)), shape=(n_vertices, n_vertices)
    ).tocsr()
    graph.sum_duplicates()
    row_sum = np.asarray(graph.sum(axis=1)).ravel()
    return sparse.diags(1.0 / np.maximum(row_sum, 1.0)) @ graph


def smooth_mask(mask, adjacency):
    """Smooth a binary parcel on the cortical mesh without blurring the t-map."""
    score = mask.astype(np.float32)
    for _ in range(SMOOTHING_STEPS):
        score = adjacency @ score
    cleaned = score >= SMOOTHING_CUTOFF
    # Keep atlas support nearby and avoid islands created far from the parcel.
    support = (adjacency @ mask.astype(np.float32)) > 0
    return cleaned & support


def boundary_segments(mask, points, triangles):
    """Return unique mesh edges that form a parcel boundary."""
    edges = np.vstack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    edge_mask = mask[edges[:, 0]] != mask[edges[:, 1]]
    edges = np.sort(edges[edge_mask], axis=1)
    if len(edges) == 0:
        return np.empty((0, 2, 2), dtype=float)
    edges = np.unique(edges, axis=0)
    return points[edges, :2]


def label_anchor(mask, points):
    """Choose an in-parcel vertex near the flat-space centroid."""
    vertex_ids = np.flatnonzero(mask)
    parcel_points = points[vertex_ids, :2]
    center = np.median(parcel_points, axis=0)
    return parcel_points[np.argmin(np.sum((parcel_points - center) ** 2, axis=1))]


def active_stats(texture, mask):
    values = texture[mask]
    active = values[np.isfinite(values)]
    return {
        "active_vertices": int(len(active)),
        "active_percent": float(100 * len(active) / max(mask.sum(), 1)),
        "max_t": float(np.nanmax(active)) if len(active) else np.nan,
        "mean_t": float(np.nanmean(active)) if len(active) else np.nan,
    }


def robust_color_limits(texture, fixed_vmax=None):
    """Return a zero-based linear t-value range for one surface map."""
    values = texture[np.isfinite(texture) & (texture > 0)]
    if len(values) == 0:
        raise ValueError("Cannot set color limits for an empty t-map texture.")
    vmax = (
        float(fixed_vmax)
        if fixed_vmax is not None
        else float(np.nanpercentile(values, DISPLAY_VMAX_PERCENTILE))
    )
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(values))
    return 0.0, vmax


def plot_flatmap(
    texture,
    title,
    out_path,
    fixed_vmax=None,
    fsaverage=None,
    overlay_groups=None,
):
    vmin, vmax = robust_color_limits(texture, fixed_vmax=fixed_vmax)
    vertex_data = cortex.Vertex(
        texture, "fsaverage", cmap="hot", vmin=vmin, vmax=vmax
    )
    fig = cortex.quickflat.make_figure(
        vertex_data,
        with_curvature=True,
        with_colorbar=False,
        with_rois=False,
        with_sulci=False,
        with_labels=False,
        curvature_brightness=0.56,
        curvature_contrast=0.24,
        height=1050,
    )
    ax = fig.axes[0]
    if fsaverage is None:
        raise ValueError("fsaverage is required for the atlas label overlay.")
    # Draw panel-specific requested parcels regardless of individual t-map
    # support. ``None`` retains the complete shared layer for the Scene panel.
    overlay_kwargs = {
        "label_size": 35,
        "min_overlap_vertices": 1,
        "min_label_vertices": 1,
        "min_label_spacing_px": 0,
        "boundary_color": "white",
        "boundary_linewidth": 1.0,
    }
    if overlay_groups is None:
        add_requested_aal3_overlay(ax, fsaverage, **overlay_kwargs)
        add_requested_aparc_overlay(ax, fsaverage, **overlay_kwargs)
        add_requested_wang2015_overlay(ax, fsaverage, **overlay_kwargs)
        add_requested_hcp_mmp_overlay(ax, fsaverage, **overlay_kwargs)
    else:
        if "aal3" in overlay_groups:
            add_requested_aal3_overlay(
                ax, fsaverage, groups=overlay_groups["aal3"], **overlay_kwargs
            )
        if "aparc" in overlay_groups:
            add_requested_aparc_overlay(
                ax, fsaverage, groups=overlay_groups["aparc"], **overlay_kwargs
            )
        if "wang2015" in overlay_groups:
            add_requested_wang2015_overlay(
                ax, fsaverage, groups=overlay_groups["wang2015"], **overlay_kwargs
            )
        if "hcp_mmp" in overlay_groups:
            add_requested_hcp_mmp_overlay(
                ax, fsaverage, groups=overlay_groups["hcp_mmp"], **overlay_kwargs
            )

    brain_position = ax.get_position()
    ax.text(
        -0.070,
        0.5,
        title,
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=34,
        fontweight="bold",
    )
    cbar_ax = fig.add_axes(
        [
            brain_position.x1 + 0.018,
            brain_position.y0 + 0.08 * brain_position.height,
            0.026,
            0.84 * brain_position.height,
        ]
    )
    scalar_map = cm.ScalarMappable(
        cmap="hot", norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
    )
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, cax=cbar_ax, orientation="vertical")
    colorbar.set_label("t-value", fontsize=36, fontweight="bold", labelpad=18)
    colorbar.ax.tick_params(labelsize=28, width=1.3, length=6)
    colorbar.outline.set_linewidth(0.8)

    fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def combine_images_vertically(image_paths, out_path):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    width = max(image.width for image in images)

    def centered(image):
        if image.width == width:
            return image
        canvas = Image.new("RGB", (width, image.height), "white")
        canvas.paste(image, ((width - image.width) // 2, 0))
        return canvas

    images = [centered(image) for image in images]
    gap = 16
    combined_height = sum(image.height for image in images) + gap * (len(images) - 1)
    combined = Image.new("RGB", (width, combined_height), "white")
    y = 0
    for image in images:
        combined.paste(image, (0, y))
        y += image.height + gap
    combined.save(out_path, dpi=(240, 240))
    print(f"Saved {out_path}")


def main():
    plt.close("all")
    cortex.database.db = cortex.database.Database()
    PY_CORTEX_DIR.mkdir(parents=True, exist_ok=True)

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage")

    print("Projecting statistical maps...")
    clip_texture = load_tmap_texture(CLIP_TMAP, fsaverage)
    gist_texture = load_tmap_texture(GIST_TMAP, fsaverage)
    va_texture = load_tmap_texture(VA_TMAP, fsaverage)

    rows = []
    for name, texture, fixed_vmax in [
        ("clip", clip_texture, None),
        ("gist", gist_texture, None),
        ("valence_arousal", va_texture, VA_DISPLAY_VMAX),
    ]:
        values = texture[np.isfinite(texture) & (texture > 0)]
        vmin, vmax = robust_color_limits(texture, fixed_vmax=fixed_vmax)
        rows.append(
            {
                "map": name,
                "active_surface_vertices": int(len(values)),
                "min_positive_t": float(np.nanmin(values)),
                "max_positive_t": float(np.nanmax(values)),
                "color_vmin": float(vmin),
                "color_vmax": float(vmax),
                "color_vmax_percentile": (
                    DISPLAY_VMAX_PERCENTILE if fixed_vmax is None else np.nan
                ),
                "fixed_color_vmax": fixed_vmax,
            }
        )
    pd.DataFrame(rows).to_csv(OUT_SUMMARY, index=False)

    plot_flatmap(
        va_texture,
        "Emotion RSA",
        OUT_VA,
        fixed_vmax=VA_DISPLAY_VMAX,
        fsaverage=fsaverage,
        overlay_groups=EMOTION_OVERLAY_GROUPS,
    )
    plot_flatmap(
        gist_texture,
        "GIST RSA",
        OUT_GIST,
        fsaverage=fsaverage,
        overlay_groups=GIST_OVERLAY_GROUPS,
    )
    plot_flatmap(
        clip_texture,
        "Scene RSA",
        OUT_CLIP,
        fsaverage=fsaverage,
    )
    combine_images_vertically(
        [OUT_VA, OUT_GIST, OUT_CLIP], OUT_COMBINED
    )
    MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    Image.open(OUT_COMBINED).save(OUT_MANUSCRIPT_COMBINED, dpi=(240, 240))
    Image.open(OUT_COMBINED).save(OUT_MANUSCRIPT_LATEST, dpi=(240, 240))
    # Keep the generic output name pointing to the newest complete figure.
    Image.open(OUT_COMBINED).save(OUT_LATEST, dpi=(240, 240))
    print(f"Saved {OUT_MANUSCRIPT_COMBINED}")
    print(f"Saved {OUT_MANUSCRIPT_LATEST}")
    print(f"Saved {OUT_LATEST}")


if __name__ == "__main__":
    main()
