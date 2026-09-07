#!/usr/bin/env python3
"""Generate the PhysVision results figure (per-category + per-image-type bars)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Final full-set numbers (35,596 QA pairs)
categories = ["quality", "tumor_detection", "parameter", "localization"]
cat_acc = [68.6, 64.9, 23.7, 16.1]

img_types = ["sinogram", "pet_recon", "phantom"]
type_acc = [81.4, 36.9, 34.3]

RANDOM = 25.0
OVERALL = 43.7

# Colorblind-friendly palette
c_strong = "#0072B2"
c_weak = "#D55E00"
c_type = "#009E73"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

# --- Left: by category ---
ax = axes[0]
colors = [c_strong if a >= RANDOM else c_weak for a in cat_acc]
bars = ax.bar(categories, cat_acc, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(RANDOM, color="gray", linestyle="--", linewidth=1, label=f"Random ({RANDOM:.0f}%)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy by Question Category")
ax.set_ylim(0, 100)
for b, a in zip(bars, cat_acc):
    ax.text(b.get_x() + b.get_width() / 2, a + 1.5, f"{a:.1f}", ha="center", fontsize=9)
ax.legend(loc="upper right", fontsize=9)
ax.tick_params(axis="x", rotation=20)

# --- Right: by image type ---
ax = axes[1]
bars = ax.bar(img_types, type_acc, color=c_type, edgecolor="black", linewidth=0.6)
ax.axhline(RANDOM, color="gray", linestyle="--", linewidth=1, label=f"Random ({RANDOM:.0f}%)")
ax.axhline(OVERALL, color="black", linestyle=":", linewidth=1, label=f"Overall ({OVERALL:.1f}%)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy by Image Type")
ax.set_ylim(0, 100)
for b, a in zip(bars, type_acc):
    ax.text(b.get_x() + b.get_width() / 2, a + 1.5, f"{a:.1f}", ha="center", fontsize=9)
ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
out = "figure_results.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
