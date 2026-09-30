#!/usr/bin/env python3
"""Generate the PhysVision results figure: lite vs full, per category + image type."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Final full-set numbers (35,596 QA pairs)
categories = ["tumor_det.", "quality", "parameter", "localization"]
lite_cat = [64.9, 68.6, 23.7, 16.1]
full_cat = [99.8, 99.5, 83.1, 68.7]

img_types = ["sinogram", "pet_recon", "phantom"]
lite_type = [81.4, 36.9, 34.3]
full_type = [99.7, 87.4, 84.6]

RANDOM = 25.0

c_lite = "#D55E00"   # orange
c_full = "#0072B2"   # blue

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=300)

def grouped(ax, labels, lite_vals, full_vals, title):
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax.bar(x - w/2, lite_vals, w, label="lite (23M)", color=c_lite,
                edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, full_vals, w, label="full (0.9B)", color=c_full,
                edgecolor="black", linewidth=0.5)
    ax.axhline(RANDOM, color="gray", linestyle="--", linewidth=1,
               label=f"Random ({RANDOM:.0f}%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.2,
                    f"{b.get_height():.0f}", ha="center", fontsize=7)
    ax.legend(loc="lower right", fontsize=8)

grouped(axes[0], categories, lite_cat, full_cat, "Accuracy by Question Category")
grouped(axes[1], img_types, lite_type, full_type, "Accuracy by Image Type")

plt.tight_layout()
out = "figure_results.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
