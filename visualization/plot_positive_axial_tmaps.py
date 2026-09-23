#!/usr/bin/env python
"""Plot positive-only axial t-maps on the MNI template.

This keeps the plotted image and the saved NIfTI aligned: values <= 0 are set
to zero, then the positive-only map is rendered as axial slices.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import datasets, plotting
from nilearn.image import new_img_like


DEFAULT_MAP = Path(
    r"N:\Experimental_Data\yujunchen\projects\IAPS_Searchlight"
    r"\outputs\tBrainmap\final_results\tmap_beh60_p05_v2.nii.gz"
)


def positive_only_img(path: Path):
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    data = np.where(np.isfinite(data) & (data > 0), data, 0.0)
    return new_img_like(img, data, copy_header=True), data


def nonzero_cut_coords(img, n_cuts: int = 8) -> list[float]:
    """Choose evenly spaced axial cuts through nonzero voxels."""
    data = np.asarray(img.get_fdata())
    vox = np.argwhere(data > 0)
    if vox.size == 0:
        return list(np.linspace(-30, 60, n_cuts))

    affine = img.affine
    z_mm = nib.affines.apply_affine(affine, vox)[:, 2]
    lo, hi = np.percentile(z_mm, [2, 98])
    if np.isclose(lo, hi):
        lo, hi = z_mm.min(), z_mm.max()
    return [float(x) for x in np.linspace(lo, hi, n_cuts)]


def plot_positive_axial(
    input_path: Path,
    output_dir: Path | None = None,
    title: str | None = None,
    n_cuts: int = 8,
) -> tuple[Path, Path, Path]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve() if output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pos_img, data = positive_only_img(input_path)
    positive_values = data[data > 0]
    if positive_values.size == 0:
        raise ValueError(f"No positive voxels found in {input_path}")

    stem = input_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = input_path.stem

    pos_nii = output_dir / f"{stem}_positive_only.nii.gz"
    png_path = output_dir / f"{stem}_positive_only_axial_on_template.png"
    pdf_path = output_dir / f"{stem}_positive_only_axial_on_template.pdf"
    pos_img.to_filename(str(pos_nii))

    vmax = float(np.nanmax(positive_values))
    cuts = nonzero_cut_coords(pos_img, n_cuts=n_cuts)
    bg_img = datasets.load_mni152_template(resolution=2)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.titlesize": 12,
    })
    fig = plt.figure(figsize=(13.5, 3.2), dpi=300)
    display = plotting.plot_stat_map(
        pos_img,
        bg_img=bg_img,
        display_mode="z",
        cut_coords=cuts,
        threshold=1e-6,
        cmap="YlOrRd",
        black_bg=False,
        colorbar=True,
        annotate=True,
        draw_cross=False,
        symmetric_cbar=False,
        vmin=0,
        vmax=vmax,
        figure=fig,
        title=title or stem,
    )
    display.savefig(str(png_path), dpi=300)
    display.savefig(str(pdf_path), dpi=300)
    display.close()
    plt.close(fig)
    return pos_nii, png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title", default="Behavioral VA t-map, positive values")
    parser.add_argument("--n-cuts", type=int, default=8)
    args = parser.parse_args()

    pos_nii, png_path, pdf_path = plot_positive_axial(
        args.input,
        output_dir=args.output_dir,
        title=args.title,
        n_cuts=args.n_cuts,
    )
    print(f"Saved positive-only NIfTI: {pos_nii}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
