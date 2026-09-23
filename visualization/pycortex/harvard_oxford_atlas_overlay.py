"""Atlas parcel outlines and bilateral short labels for pycortex flatmaps."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import cortex
import matplotlib.patheffects as pe
from nibabel.freesurfer.io import read_annot
import numpy as np
from nilearn import surface
from scipy import sparse


# Local AAL3 v3.0 (1-mm MNI) distribution retained with this manuscript project.
AAL3_DIR = Path(
    "/mnt/c/Users/yujunchen/OneDrive - University of Florida/"
    "BME departmental/manuscript/Emotion Semantic Categorization/"
    "tmp/nilearn_data_aal/aal_3v2/AAL3"
)
AAL3_IMAGE = AAL3_DIR / "AAL3v1_1mm.nii.gz"
AAL3_LABELS = AAL3_DIR / "AAL3v1_1mm.xml"

# The same short AAL3 name is placed in each hemisphere.  The atlas has no
# lateral-occipital or sulcal parcels, so its native gyral names are retained.
CORE_DISPLAY_REGIONS = (
    ("Precuneus", "PCUN"),
    ("Parietal_Sup", "SPL"),
    ("Cuneus", "CUN"),
    ("Angular", "ANG"),
    ("Occipital_Mid", "MOG"),
    ("Occipital_Inf", "IOG"),
    ("Fusiform", "FFG"),
    ("Temporal_Mid", "MTG"),
    ("Temporal_Inf", "ITG"),
    ("Lingual", "LING"),
)

# Major cortical parcels added for figures that need broader anatomical context.
# Smaller parcels are outlined but only labelled when there is enough surface
# area and room to keep the flatmap legible.
EXTENDED_DISPLAY_REGIONS = CORE_DISPLAY_REGIONS + (
    ("Occipital_Sup", "SOG"),
    ("Calcarine", "CAL"),
    ("Parietal_Inf", "IPL"),
    ("Postcentral", "PoCG"),
    ("Precentral", "PreCG"),
    ("Temporal_Sup", "STG"),
    ("Frontal_Mid_2", "MFG"),
    ("Frontal_Sup_2", "SFG"),
)

# All bilateral cortical AAL3 parcels that can be represented on the cortical
# flatmap.  The extended set is deliberately kept first, so the anatomical
# regions central to the manuscript result retain priority when label anchors
# would otherwise collide.  The remaining parcels are still outlined and are
# labelled whenever there is sufficient surface area and room.
ALL_CORTICAL_DISPLAY_REGIONS = EXTENDED_DISPLAY_REGIONS + (
    ("Frontal_Inf_Oper", "IFGop"),
    ("Frontal_Inf_Tri", "IFGtri"),
    ("Frontal_Inf_Orb_2", "IFGorb"),
    ("Rolandic_Oper", "ROL"),
    ("Supp_Motor_Area", "SMA"),
    ("Olfactory", "OLF"),
    ("Frontal_Sup_Medial", "mSFG"),
    ("Frontal_Med_Orb", "mOFC"),
    ("Rectus", "REC"),
    ("OFCmed", "OFCmed"),
    ("OFCant", "OFCant"),
    ("OFCpost", "OFCpost"),
    ("OFClat", "OFClat"),
    ("Insula", "INS"),
    ("Cingulate_Mid", "MCC"),
    ("Cingulate_Post", "PCC"),
    ("Hippocampus", "HPC"),
    ("ParaHippocampal", "PHG"),
    ("Amygdala", "AMY"),
    ("SupraMarginal", "SMG"),
    ("Paracentral_Lobule", "PCL"),
    ("Heschl", "HES"),
    ("Temporal_Pole_Sup", "TPsup"),
    ("Temporal_Pole_Mid", "TPmid"),
)

# AAL3 cortical parcels requested for the manuscript surface figures.  The
# display fields retain the names supplied in ``aal3_roi_name.txt``.  Thalamus
# is subcortical and therefore cannot be rendered on this cortical flatmap.
MANUSCRIPT_AAL3_DISPLAY_REGIONS = (
    ("Precentral", "Precentral"),
    ("Frontal_Sup_2", "Front Sup"),
    ("Frontal_Mid_2", "Front Mid"),
    ("Frontal_Inf_Oper", "Front Inf Oper"),
    ("Frontal_Inf_Tri", "Front Inf Tri"),
    ("Frontal_Inf_Orb_2", "Front Inf Orb"),
    ("Supp_Motor_Area", "SMA"),
    ("Frontal_Sup_Medial", "Front Sup Medial"),
    ("Frontal_Med_Orb", "Front Med Orb"),
    ("OFCmed", "OFCmed"),
    ("OFCant", "OFCant"),
    ("OFCpost", "OFCpost"),
    ("OFClat", "OFClat"),
    ("Insula", "Insula"),
    ("Cingulate_Mid", "Cingulate Mid"),
    ("Cingulate_Post", "Cingulate Post"),
    ("Hippocampus", "Hippocampus"),
    ("ParaHippocampal", "ParaHippocampal"),
    ("Amygdala", "Amygdala"),
    ("Calcarine", "Calcarine"),
    ("Cuneus", "Cuneus"),
    ("Lingual", "Lingual"),
    ("Occipital_Sup", "Occ Sup"),
    ("Occipital_Mid", "Occ Mid"),
    ("Occipital_Inf", "Occ Inf"),
    ("Fusiform", "Fusiform"),
    ("Postcentral", "Postcentral"),
    ("SupraMarginal", "SupraMarginal"),
    ("Angular", "Angular"),
    ("Precuneus", "Precuneus"),
    ("Temporal_Sup", "Temp Sup"),
    ("Temporal_Pole_Sup", "Temp Pole Sup"),
    ("Temporal_Mid", "Temp Mid"),
    ("Temporal_Pole_Mid", "Temp Pole Mid"),
    ("Temporal_Inf", "Temp Inf"),
    ("ACC_sub", "ACC sub"),
    ("ACC_pre", "ACC pre"),
    ("ACC_sup", "ACC sup"),
)

OUTLINE_COLORS = (
    "#E76F51", "#F4A261", "#E9C46A", "#2A9D8F", "#22A7B8",
    "#3A86C8", "#8E7DBE", "#B07AA1", "#78A6A3", "#D17A9B",
    "#6C8EAD", "#B48E58", "#7B8E4B", "#B86F93", "#617C64",
    "#A8795D", "#6B7FA1", "#8A6A9E",
)
BOUNDARY_SMOOTHING_STEPS = 4
BOUNDARY_LINEWIDTH = 2.4

# Atlas groups used for the manuscript flatmap requested in September 2026.
# Each item is a display label followed by the bilateral AAL3 source parcels.
REQUESTED_AAL3_GROUPS = (
    ("SMA", ("Supp_Motor_Area",)),
    ("OFC", ("Frontal_Med_Orb", "OFCmed", "OFCant", "OFCpost", "OFClat")),
    ("Insula", ("Insula",)),
)

# HCP--MMP labels provide the locally available surface representations for the
# requested functional ROIs.
HCP_MMP_DIR = Path(
    "/mnt/c/Users/yujunchen/OneDrive - University of Florida/"
    "BME departmental/manuscript/Emotion Semantic Categorization/tmp/hcp_mmp"
)
HCP_MMP_ANNOTATIONS = {
    "L": HCP_MMP_DIR / "lh.HCP-MMP1.annot",
    "R": HCP_MMP_DIR / "rh.HCP-MMP1.annot",
}
REQUESTED_HCP_MMP_GROUPS = (
    ("RSC", ("RSC",)),
)

# The prior flatmap used FreeSurfer's fsaverage ``aparc`` (Desikan--Killiany)
# surface parcellation for these two anatomical labels. Retain those exact
# parcels rather than substituting AAL3 SupraMarginal or the HCP--MMP STS areas.
FSAVERAGE_APARC_DIR = Path(
    "/usr/local/freesurfer/7.4.1/subjects/fsaverage/label"
)
FSAVERAGE_APARC_ANNOTATIONS = {
    "L": FSAVERAGE_APARC_DIR / "lh.aparc.annot",
    "R": FSAVERAGE_APARC_DIR / "rh.aparc.annot",
}
REQUESTED_APARC_GROUPS = (
    ("STS", ("bankssts",)),
    ("SMG", ("supramarginal",)),
)

# Wang et al. (2015; commonly called the Kastner visual atlas) uses the IDs
# below in its MNI-space maximum-probability maps. The requested visual-field
# labels deliberately combine the submaps named in the manuscript request.
WANG2015_DIR = Path(
    "/mnt/c/Users/yujunchen/OneDrive - University of Florida/"
    "BME departmental/manuscript/Emotion Semantic Categorization/tmp/wang2015/"
    "subj_vol_all"
)
WANG2015_IMAGES = {
    "L": WANG2015_DIR / "maxprob_vol_lh.nii.gz",
    "R": WANG2015_DIR / "maxprob_vol_rh.nii.gz",
}
WANG2015_PROBABILITY_MINIMUM = 10.0
REQUESTED_WANG2015_GROUPS = (
    ("V1", (1, 2)),
    ("V2", (3, 4)),
    ("V3", (5, 6)),
    ("V4", (7,)),
    ("VO", (8, 9)),
    ("LO", (14, 15)),
    ("MT", (12, 13)),
    ("IPS", (18, 19, 20, 21, 22, 23)),
    ("SPL", (24,)),
    ("FEF", (25,)),
)


def _read_aal3_labels() -> dict[str, int]:
    """Read AAL3 label names and their integer voxel IDs from its XML file."""
    root = ET.parse(AAL3_LABELS).getroot()
    return {
        label.findtext("name"): int(label.findtext("index"))
        for label in root.findall("./data/label")
    }


def _project_atlas(fsaverage) -> np.ndarray:
    left = surface.vol_to_surf(
        AAL3_IMAGE,
        fsaverage.pial_left,
        inner_mesh=fsaverage.white_left,
        interpolation="nearest_most_frequent",
        n_samples=9,
    )
    right = surface.vol_to_surf(
        AAL3_IMAGE,
        fsaverage.pial_right,
        inner_mesh=fsaverage.white_right,
        interpolation="nearest_most_frequent",
        n_samples=9,
    )
    return np.nan_to_num(np.hstack([left, right]), nan=0).astype(np.int16)


def _normalized_mesh_adjacency(n_vertices: int, triangles: np.ndarray):
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


def _smoothed_mask(mask: np.ndarray, adjacency) -> np.ndarray:
    score = mask.astype(np.float32)
    for _ in range(BOUNDARY_SMOOTHING_STEPS):
        score = adjacency @ score
    return score


def _label_anchor(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    vertices = np.flatnonzero(mask)
    parcel_points = points[vertices, :2]
    center = np.median(parcel_points, axis=0)
    return parcel_points[np.argmin(np.sum((parcel_points - center) ** 2, axis=1))]


def _separated_label_anchor(
    anchor: np.ndarray,
    parcel_points: np.ndarray,
    ax,
    placed_positions: list[np.ndarray],
    minimum_spacing_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Move a label within its parcel when needed, while retaining every label."""
    anchor_px = ax.transData.transform(anchor)
    if minimum_spacing_px <= 0:
        return anchor, anchor_px

    # Consider only mesh vertices in the same parcel. Pixel-space distances
    # keep text spacing stable while ensuring labels never drift off an ROI.
    candidate_px = ax.transData.transform(parcel_points)
    if placed_positions:
        distances = np.linalg.norm(
            candidate_px[:, None, :] - np.asarray(placed_positions)[None, :, :],
            axis=2,
        )
        clearance = distances.min(axis=1)
    else:
        clearance = np.full(len(parcel_points), np.inf)
    distance_from_anchor = np.linalg.norm(candidate_px - anchor_px, axis=1)
    # The medial occipital maps can occupy an unusually long strip after the
    # cortical cut is flattened.  Do not let collision avoidance send their
    # text to a distant end of that strip: labels should remain near the
    # anatomical centre of the parcel even when perfect text separation is not
    # possible.  The displacement cap is deliberately smaller than the
    # requested label-to-label spacing, so it is a local adjustment only.
    maximum_displacement_px = min(60.0, max(30.0, minimum_spacing_px * 0.45))
    nearby = distance_from_anchor <= maximum_displacement_px
    available = (clearance >= minimum_spacing_px) & nearby
    if available.any():
        index = np.flatnonzero(available)[np.argmin(distance_from_anchor[available])]
    else:
        # A compact or elongated parcel can lack a fully separated position.
        # Keep its label local to its anatomical anchor and choose the locally
        # clearest in-parcel point rather than moving it to a remote segment.
        local_indices = np.flatnonzero(nearby)
        if local_indices.size:
            index = local_indices[np.argmax(clearance[local_indices])]
        else:
            index = int(np.argmin(distance_from_anchor))
    return parcel_points[index], candidate_px[index]


def add_harvard_oxford_overlay(
    ax,
    fsaverage,
    *,
    label_size: float = 15.0,
    include_extended_regions: bool = False,
    include_all_cortical_regions: bool = False,
    display_regions: tuple[tuple[str, str], ...] | None = None,
    roi_mask: np.ndarray | None = None,
    min_overlap_vertices: int = 1,
    min_label_vertices: int = 450,
    min_label_spacing_px: float = 115.0,
    boundary_color: str | None = None,
    boundary_linewidth: float | None = None,
):
    """Overlay bilateral AAL3 parcel boundaries and concise names on a flatmap."""
    if not AAL3_IMAGE.exists() or not AAL3_LABELS.exists():
        raise FileNotFoundError(f"AAL3 atlas files are missing from {AAL3_DIR}")

    atlas_texture = _project_atlas(fsaverage)
    label_ids = _read_aal3_labels()
    points, triangles = cortex.db.get_surf("fsaverage", "flat", merge=True, nudge=True)
    if atlas_texture.size != points.shape[0]:
        raise ValueError("AAL3 projection and pycortex flat surface have different vertex counts.")
    if roi_mask is not None and roi_mask.shape != atlas_texture.shape:
        raise ValueError("ROI overlap mask and AAL3 surface texture have different shapes.")

    adjacency = _normalized_mesh_adjacency(atlas_texture.size, triangles)
    left_vertices = surface.load_surf_mesh(fsaverage.pial_left).coordinates.shape[0]
    hemisphere_masks = (
        np.arange(atlas_texture.size) < left_vertices,
        np.arange(atlas_texture.size) >= left_vertices,
    )
    uses_explicit_display_regions = display_regions is not None
    if display_regions is not None:
        pass
    elif include_all_cortical_regions:
        display_regions = ALL_CORTICAL_DISPLAY_REGIONS
    elif include_extended_regions:
        display_regions = EXTENDED_DISPLAY_REGIONS
    else:
        display_regions = CORE_DISPLAY_REGIONS
    placed_label_positions = []
    included_labels = []
    for index, (base_name, short_name) in enumerate(display_regions):
        color = boundary_color or OUTLINE_COLORS[index % len(OUTLINE_COLORS)]
        bilateral_ids = (label_ids[f"{base_name}_L"], label_ids[f"{base_name}_R"])
        for label_id, hemisphere_mask in zip(bilateral_ids, hemisphere_masks):
            mask = (atlas_texture == label_id) & hemisphere_mask
            if not mask.any():
                continue
            overlap_mask = mask if roi_mask is None else mask & roi_mask
            if overlap_mask.sum() < min_overlap_vertices:
                continue
            smoothed = _smoothed_mask(mask, adjacency)
            ax.tricontour(
                points[:, 0],
                points[:, 1],
                triangles,
                smoothed,
                levels=[0.5],
                colors=[color],
                linewidths=(
                    BOUNDARY_LINEWIDTH
                    if boundary_linewidth is None
                    else boundary_linewidth
                ),
                alpha=0.98,
                zorder=7,
            )
            if overlap_mask.sum() < min_label_vertices:
                continue
            anchor = _label_anchor(overlap_mask, points)
            anchor_px = ax.transData.transform(anchor)
            if any(
                np.linalg.norm(anchor_px - previous_px) < min_label_spacing_px
                for previous_px in placed_label_positions
            ):
                continue
            ax.text(
                anchor[0],
                anchor[1],
                short_name,
                color="white",
                fontsize=label_size,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=8,
                path_effects=[pe.withStroke(linewidth=3.0, foreground="black")],
            )
            placed_label_positions.append(anchor_px)
            included_labels.append(short_name)

    return {
        "atlas": "AAL3 v3.0 (1 mm MNI)",
        "bilateral_labels": {
            base_name: short_name for base_name, short_name in display_regions
        },
        "include_extended_regions": include_extended_regions,
        "include_all_cortical_regions": include_all_cortical_regions,
        "display_regions_source": (
            "explicit" if uses_explicit_display_regions else "built-in selection"
        ),
        "restricted_to_roi_overlap": roi_mask is not None,
        "min_overlap_vertices": min_overlap_vertices,
        "min_label_vertices": min_label_vertices,
        "min_label_spacing_px": min_label_spacing_px,
        "boundary_color": boundary_color,
        "boundary_linewidth": boundary_linewidth,
        "visible_labels": included_labels,
    }


def _draw_group_overlay(
    ax,
    points: np.ndarray,
    triangles: np.ndarray,
    adjacency,
    hemisphere_masks: tuple[np.ndarray, np.ndarray],
    groups: tuple[tuple[str, tuple[str, ...]], ...],
    group_masks: dict[tuple[str, str], np.ndarray],
    *,
    atlas_name: str,
    label_size: float,
    roi_mask: np.ndarray | None,
    min_overlap_vertices: int,
    min_label_vertices: int,
    min_label_spacing_px: float,
    boundary_color: str,
    boundary_linewidth: float,
):
    """Draw smoothed, bilateral parcel-group outlines and white outlined labels."""
    placed_label_positions = []
    visible_labels = []
    label_entries = []
    for display_name, _ in groups:
        for hemisphere, hemisphere_mask in zip(("L", "R"), hemisphere_masks):
            mask = group_masks[(display_name, hemisphere)] & hemisphere_mask
            if not mask.any():
                continue
            overlap_mask = mask if roi_mask is None else mask & roi_mask
            if overlap_mask.sum() < min_overlap_vertices:
                continue
            smoothed = _smoothed_mask(mask, adjacency)
            ax.tricontour(
                points[:, 0], points[:, 1], triangles, smoothed,
                levels=[0.5], colors=[boundary_color], linewidths=boundary_linewidth,
                alpha=0.98, zorder=7,
            )
            if overlap_mask.sum() < min_label_vertices:
                continue
            label_entries.append((display_name, hemisphere, mask, overlap_mask))

    # Place compact parcels first. This lets a broad combined ROI (for example
    # LO or IPS) choose an alternative in-parcel anchor around smaller nearby
    # areas instead of forcing their labels to collide.
    for display_name, hemisphere, mask, overlap_mask in sorted(
        label_entries, key=lambda entry: int(entry[2].sum())
    ):
        anchor = _label_anchor(overlap_mask, points)
        label_anchor, anchor_px = _separated_label_anchor(
            anchor, points[mask, :2], ax, placed_label_positions,
            min_label_spacing_px,
        )
        ax.text(
            label_anchor[0], label_anchor[1], display_name, color="white",
            fontsize=label_size, fontweight="bold", ha="center", va="center",
            zorder=8,
            path_effects=[pe.withStroke(linewidth=3.0, foreground="black")],
        )
        placed_label_positions.append(anchor_px)
        visible_labels.append(f"{display_name}_{hemisphere}")
    return {"atlas": atlas_name, "visible_labels": visible_labels}


def add_requested_aal3_overlay(
    ax,
    fsaverage,
    *,
    groups: tuple[tuple[str, tuple[str, ...]], ...] = REQUESTED_AAL3_GROUPS,
    label_size: float = 30.0,
    roi_mask: np.ndarray | None = None,
    min_overlap_vertices: int = 1,
    min_label_vertices: int = 1,
    min_label_spacing_px: float = 0.0,
    boundary_color: str = "white",
    boundary_linewidth: float = 1.0,
):
    """Overlay the requested combined bilateral AAL3 anatomical ROIs."""
    if not AAL3_IMAGE.exists() or not AAL3_LABELS.exists():
        raise FileNotFoundError(f"AAL3 atlas files are missing from {AAL3_DIR}")
    atlas_texture = _project_atlas(fsaverage)
    labels = _read_aal3_labels()
    points, triangles = cortex.db.get_surf("fsaverage", "flat", merge=True, nudge=True)
    if atlas_texture.size != points.shape[0]:
        raise ValueError("AAL3 projection and pycortex flat surface have different shapes.")
    adjacency = _normalized_mesh_adjacency(atlas_texture.size, triangles)
    left_count = surface.load_surf_mesh(fsaverage.pial_left).coordinates.shape[0]
    hemispheres = (
        np.arange(atlas_texture.size) < left_count,
        np.arange(atlas_texture.size) >= left_count,
    )
    masks = {}
    for display_name, source_names in groups:
        for hemisphere in ("L", "R"):
            source_ids = [labels[f"{name}_{hemisphere}"] for name in source_names]
            masks[(display_name, hemisphere)] = np.isin(atlas_texture, source_ids)
    result = _draw_group_overlay(
        ax, points, triangles, adjacency, hemispheres, groups, masks,
        atlas_name="AAL3 v3.0 (1 mm MNI)", label_size=label_size,
        roi_mask=roi_mask, min_overlap_vertices=min_overlap_vertices,
        min_label_vertices=min_label_vertices,
        min_label_spacing_px=min_label_spacing_px,
        boundary_color=boundary_color, boundary_linewidth=boundary_linewidth,
    )
    result.update({
        "groups": {name: list(source_names) for name, source_names in groups},
        "boundary_smoothing_steps": BOUNDARY_SMOOTHING_STEPS,
        "boundary_color": boundary_color,
        "boundary_linewidth": boundary_linewidth,
        "label_size": label_size,
    })
    return result


def add_requested_hcp_mmp_overlay(
    ax,
    fsaverage,
    *,
    groups: tuple[tuple[str, tuple[str, ...]], ...] = REQUESTED_HCP_MMP_GROUPS,
    label_size: float = 30.0,
    roi_mask: np.ndarray | None = None,
    min_overlap_vertices: int = 1,
    min_label_vertices: int = 1,
    min_label_spacing_px: float = 0.0,
    boundary_color: str = "white",
    boundary_linewidth: float = 1.0,
):
    """Overlay bilateral HCP--MMP functional-ROI correspondences on fsaverage."""
    missing = [path for path in HCP_MMP_ANNOTATIONS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"HCP--MMP annotation files are missing: {missing}")
    points, triangles = cortex.db.get_surf("fsaverage", "flat", merge=True, nudge=True)
    left_labels, _, left_names = read_annot(HCP_MMP_ANNOTATIONS["L"])
    right_labels, _, right_names = read_annot(HCP_MMP_ANNOTATIONS["R"])
    if points.shape[0] != left_labels.size + right_labels.size:
        raise ValueError("HCP--MMP annotations and pycortex flat surface have different shapes.")
    adjacency = _normalized_mesh_adjacency(points.shape[0], triangles)
    hemispheres = (
        np.arange(points.shape[0]) < left_labels.size,
        np.arange(points.shape[0]) >= left_labels.size,
    )
    labels_by_hemisphere = {"L": left_labels, "R": right_labels}
    names_by_hemisphere = {"L": left_names, "R": right_names}
    masks = {}
    for display_name, source_names in groups:
        for hemisphere in ("L", "R"):
            decoded = [name.decode().removeprefix(f"{hemisphere}_").removesuffix("_ROI")
                       for name in names_by_hemisphere[hemisphere]]
            source_ids = [decoded.index(name) for name in source_names]
            local_mask = np.isin(labels_by_hemisphere[hemisphere], source_ids)
            if hemisphere == "L":
                masks[(display_name, hemisphere)] = np.hstack(
                    [local_mask, np.zeros(right_labels.size, dtype=bool)]
                )
            else:
                masks[(display_name, hemisphere)] = np.hstack(
                    [np.zeros(left_labels.size, dtype=bool), local_mask]
                )
    result = _draw_group_overlay(
        ax, points, triangles, adjacency, hemispheres, groups, masks,
        atlas_name="HCP--MMP1.0 surface atlas", label_size=label_size,
        roi_mask=roi_mask, min_overlap_vertices=min_overlap_vertices,
        min_label_vertices=min_label_vertices,
        min_label_spacing_px=min_label_spacing_px,
        boundary_color=boundary_color, boundary_linewidth=boundary_linewidth,
    )
    result.update({
        "groups": {name: list(source_names) for name, source_names in groups},
        "boundary_smoothing_steps": BOUNDARY_SMOOTHING_STEPS,
        "boundary_color": boundary_color,
        "boundary_linewidth": boundary_linewidth,
        "label_size": label_size,
    })
    return result


def add_requested_aparc_overlay(
    ax,
    fsaverage,
    *,
    groups: tuple[tuple[str, tuple[str, ...]], ...] = REQUESTED_APARC_GROUPS,
    label_size: float = 30.0,
    roi_mask: np.ndarray | None = None,
    min_overlap_vertices: int = 1,
    min_label_vertices: int = 1,
    min_label_spacing_px: float = 0.0,
    boundary_color: str = "white",
    boundary_linewidth: float = 1.0,
):
    """Overlay the prior FreeSurfer aparc STS and SMG parcel definitions."""
    missing = [path for path in FSAVERAGE_APARC_ANNOTATIONS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"FreeSurfer aparc annotation files are missing: {missing}")
    points, triangles = cortex.db.get_surf("fsaverage", "flat", merge=True, nudge=True)
    left_labels, _, left_names = read_annot(FSAVERAGE_APARC_ANNOTATIONS["L"])
    right_labels, _, right_names = read_annot(FSAVERAGE_APARC_ANNOTATIONS["R"])
    if points.shape[0] != left_labels.size + right_labels.size:
        raise ValueError("FreeSurfer aparc annotations and pycortex flat surface have different shapes.")
    adjacency = _normalized_mesh_adjacency(points.shape[0], triangles)
    hemispheres = (
        np.arange(points.shape[0]) < left_labels.size,
        np.arange(points.shape[0]) >= left_labels.size,
    )
    labels_by_hemisphere = {"L": left_labels, "R": right_labels}
    names_by_hemisphere = {"L": left_names, "R": right_names}
    masks = {}
    for display_name, source_names in groups:
        for hemisphere in ("L", "R"):
            decoded = [name.decode() if isinstance(name, bytes) else name
                       for name in names_by_hemisphere[hemisphere]]
            source_ids = [decoded.index(name) for name in source_names]
            local_mask = np.isin(labels_by_hemisphere[hemisphere], source_ids)
            if hemisphere == "L":
                masks[(display_name, hemisphere)] = np.hstack(
                    [local_mask, np.zeros(right_labels.size, dtype=bool)]
                )
            else:
                masks[(display_name, hemisphere)] = np.hstack(
                    [np.zeros(left_labels.size, dtype=bool), local_mask]
                )
    result = _draw_group_overlay(
        ax, points, triangles, adjacency, hemispheres, groups, masks,
        atlas_name="FreeSurfer fsaverage aparc (Desikan--Killiany)",
        label_size=label_size, roi_mask=roi_mask,
        min_overlap_vertices=min_overlap_vertices,
        min_label_vertices=min_label_vertices,
        min_label_spacing_px=min_label_spacing_px,
        boundary_color=boundary_color, boundary_linewidth=boundary_linewidth,
    )
    result.update({
        "groups": {name: list(source_names) for name, source_names in groups},
        "boundary_smoothing_steps": BOUNDARY_SMOOTHING_STEPS,
        "boundary_color": boundary_color,
        "boundary_linewidth": boundary_linewidth,
        "label_size": label_size,
    })
    return result


def add_requested_wang2015_overlay(
    ax,
    fsaverage,
    *,
    groups: tuple[tuple[str, tuple[int, ...]], ...] = REQUESTED_WANG2015_GROUPS,
    label_size: float = 30.0,
    roi_mask: np.ndarray | None = None,
    min_overlap_vertices: int = 1,
    min_label_vertices: int = 1,
    min_label_spacing_px: float = 0.0,
    boundary_color: str = "white",
    boundary_linewidth: float = 1.0,
):
    """Overlay combined Wang--Kastner (2015) visual field maps on fsaverage."""
    missing = [path for path in WANG2015_IMAGES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Wang--Kastner visual atlas files are missing: {missing}")
    left_texture = surface.vol_to_surf(
        WANG2015_IMAGES["L"], fsaverage.pial_left,
        inner_mesh=fsaverage.white_left, interpolation="nearest_most_frequent",
        n_samples=9,
    )
    right_texture = surface.vol_to_surf(
        WANG2015_IMAGES["R"], fsaverage.pial_right,
        inner_mesh=fsaverage.white_right, interpolation="nearest_most_frequent",
        n_samples=9,
    )
    atlas_texture = np.nan_to_num(
        np.hstack([left_texture, right_texture]), nan=0
    ).astype(np.int16)
    points, triangles = cortex.db.get_surf("fsaverage", "flat", merge=True, nudge=True)
    if points.shape[0] != atlas_texture.size:
        raise ValueError("Wang--Kastner atlas and pycortex flat surface have different shapes.")
    adjacency = _normalized_mesh_adjacency(atlas_texture.size, triangles)
    left_count = left_texture.size
    hemispheres = (
        np.arange(atlas_texture.size) < left_count,
        np.arange(atlas_texture.size) >= left_count,
    )
    masks = {}
    probability_backed_groups = []
    for display_name, source_ids in groups:
        for hemisphere, hemisphere_mask in zip(("L", "R"), hemispheres):
            mask = (
                np.isin(atlas_texture, source_ids) & hemisphere_mask
            )
            # The maximum-probability volume has no surviving label in a few
            # small maps (notably SPL and left FEF). Recover those outlines
            # from the supplied per-area probability images instead of silently
            # omitting a requested atlas region.
            if not mask.any():
                mesh = fsaverage.pial_left if hemisphere == "L" else fsaverage.pial_right
                white = fsaverage.white_left if hemisphere == "L" else fsaverage.white_right
                probability_mask = np.zeros(left_count if hemisphere == "L" else right_texture.size, dtype=bool)
                suffix = "lh" if hemisphere == "L" else "rh"
                for source_id in source_ids:
                    probability_path = WANG2015_DIR / f"perc_VTPM_vol_roi{source_id}_{suffix}.nii.gz"
                    texture = surface.vol_to_surf(
                        probability_path, mesh, inner_mesh=white,
                        interpolation="linear", n_samples=9,
                    )
                    probability_mask |= np.nan_to_num(texture, nan=0.0) >= WANG2015_PROBABILITY_MINIMUM
                if hemisphere == "L":
                    mask = np.hstack([probability_mask, np.zeros(right_texture.size, dtype=bool)])
                else:
                    mask = np.hstack([np.zeros(left_count, dtype=bool), probability_mask])
                probability_backed_groups.append(f"{display_name}_{hemisphere}")
            masks[(display_name, hemisphere)] = mask
    result = _draw_group_overlay(
        ax, points, triangles, adjacency, hemispheres, groups, masks,
        atlas_name="Wang et al. 2015 probabilistic visual-topography atlas",
        label_size=label_size, roi_mask=roi_mask,
        min_overlap_vertices=min_overlap_vertices,
        min_label_vertices=min_label_vertices,
        min_label_spacing_px=min_label_spacing_px,
        boundary_color=boundary_color, boundary_linewidth=boundary_linewidth,
    )
    result.update({
        "groups": {name: list(source_ids) for name, source_ids in groups},
        "boundary_smoothing_steps": BOUNDARY_SMOOTHING_STEPS,
        "boundary_color": boundary_color,
        "boundary_linewidth": boundary_linewidth,
        "label_size": label_size,
        "probability_fallback_groups": probability_backed_groups,
        "probability_fallback_minimum_percent": WANG2015_PROBABILITY_MINIMUM,
    })
    return result
