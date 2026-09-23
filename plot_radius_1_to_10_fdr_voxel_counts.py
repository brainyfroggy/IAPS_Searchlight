from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"N:\Experimental_Data\yujunchen\projects\IAPS_Searchlight")
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "tBrainmap"
SUMMARY_1_TO_5 = OUTPUT_ROOT / "behavioral_radius_1_to_5" / "behavioral_radius_1_to_5_summary.csv"
SUMMARY_6_TO_10 = OUTPUT_ROOT / "behavioral_radius_6_to_10" / "behavioral_radius_6_to_10_summary.csv"
PLOT_DIR = OUTPUT_ROOT / "behavioral_radius_1_to_10"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


summary = pd.concat(
    [pd.read_csv(SUMMARY_1_TO_5), pd.read_csv(SUMMARY_6_TO_10)],
    ignore_index=True,
    sort=False,
).sort_values("radius_voxels")

plot_data = summary[
    [
        "radius_voxels",
        "radius_mm",
        "status",
        "n_fdr_positive_voxels_before_cluster",
        "n_voxels_after_cluster_threshold",
    ]
].copy()
plot_data.to_csv(PLOT_DIR / "radius_1_to_10_fdr_cluster30_voxel_counts.csv", index=False)

valid = plot_data[plot_data["n_voxels_after_cluster_threshold"].notna()].copy()
x = valid["radius_mm"].to_numpy(dtype=float)
y = valid["n_voxels_after_cluster_threshold"].to_numpy(dtype=int)

plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
ax.plot(x, y, color="#1769aa", marker="o", linewidth=2, markersize=5)

for radius_mm, count in zip(x, y):
    ax.annotate(
        f"{count:,}",
        (radius_mm, count),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
    )

ax.scatter([3], [0], marker="x", s=55, linewidths=1.8, color="#666666", zorder=3)
ax.annotate(
    "undefined\n(one voxel)",
    (3, 0),
    xytext=(7, 13),
    textcoords="offset points",
    ha="left",
    va="bottom",
    fontsize=8,
    color="#555555",
)

ax.set_title("Searchlight-radius sensitivity in analysis space")
ax.set_xlabel("Searchlight radius (mm; resampled 3-mm isotropic grid)")
ax.set_ylabel("Voxels after FDR and cluster threshold (≥30 voxels)")
ax.set_xticks(np.arange(3, 31, 3))
ax.set_xlim(2, 31)
ax.set_ylim(0, 1200)
ax.grid(axis="y", color="#d0d0d0", linewidth=0.7, alpha=0.7)
ax.spines[["top", "right"]].set_visible(False)

fig.text(
    0.5,
    -0.015,
    "Functional images were acquired at 3.5-mm isotropic resolution and resampled to 3-mm isotropic space for analysis.",
    ha="center",
    va="top",
    fontsize=8,
)

png_path = PLOT_DIR / "radius_1_to_10_fdr_cluster30_analysis_grid.png"
svg_path = PLOT_DIR / "radius_1_to_10_fdr_cluster30_analysis_grid.svg"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.close(fig)

print(plot_data.to_string(index=False))
print(f"Saved: {png_path}")
print(f"Saved: {svg_path}")
