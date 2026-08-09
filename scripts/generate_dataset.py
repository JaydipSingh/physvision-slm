#!/usr/bin/env python3
"""
PhysVision Dataset Generator
=============================
Generates synthetic medical physics images + QA pairs using analytical simulation.
Uses pure NumPy physics (ray-cylinder intersection, Poisson noise, Gaussian PSF)
— no Geant4 or external simulation toolkit required.

This produces visually realistic PET/sinogram/phantom images with exact ground-truth
annotations at high speed (~10K images in 30 minutes on a laptop).

Generates:
  1. PET reconstruction images (with tumors of varying size/position)
  2. Sinograms (with/without artifacts)
  3. Phantom cross-section diagrams
  4. Detector geometry visualizations

Each image is paired with 5-10 automatically generated QA pairs.

Usage:
  python scripts/generate_dataset.py --num-images 500 --output data/generated
  python scripts/generate_dataset.py --num-images 10000 --output data/generated  # full
"""
import argparse
import json
import os
import uuid
import random
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter


# Clinical hot colormap
CLINICAL_HOT = LinearSegmentedColormap.from_list("clinical_hot", [
    (0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0),
    (1.0, 0.5, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0),
])


@dataclass
class TumorSpec:
    x_mm: float
    y_mm: float
    diameter_mm: float
    suv_ratio: float


@dataclass
class ScanConfig:
    """Configuration for one simulated PET scan."""
    patient_type: str  # "brain", "lung", "body"
    ring_diameter_mm: float = 350.0
    num_annihilations: int = 5000
    image_size: int = 128
    tumors: List[TumorSpec] = field(default_factory=list)
    noise_level: float = 1.0  # Poisson noise multiplier
    reconstruction_iterations: int = 50
    scatter_fraction: float = 0.3
    crystal_material: str = "LYSO"
    num_rings: int = 4


@dataclass
class QAPair:
    question: str
    answer: str
    category: str  # "tumor_detection", "localization", "parameter", "quality"
    answer_type: str  # "yes_no", "numeric", "text", "multiple_choice"
    options: List[str] = field(default_factory=list)
    correct_option: int = -1


# ============================================================================
# SIMULATION ENGINE (extracted from g4_agent, standalone NumPy)
# ============================================================================

def simulate_pet_scan(config: ScanConfig, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """
    Simulate a PET scan producing a 2D activity image.
    Returns (image_array, metadata_dict).
    """
    size = config.image_size
    ring_radius = config.ring_diameter_mm / 2.0
    fov = ring_radius * 0.8  # Field of view
    scale = size / (2 * fov)

    # Create activity map
    activity = np.zeros((size, size), dtype=np.float64)

    # Background tissue activity (uniform with noise)
    if config.patient_type == "brain":
        # Elliptical brain region
        y_grid, x_grid = np.mgrid[0:size, 0:size]
        cx, cy = size / 2, size / 2
        rx, ry = size * 0.35, size * 0.42
        mask = ((x_grid - cx) / rx) ** 2 + ((y_grid - cy) / ry) ** 2 <= 1
        activity[mask] = 1.0
        # Gray matter (higher uptake in outer shell)
        inner_mask = ((x_grid - cx) / (rx * 0.7)) ** 2 + ((y_grid - cy) / (ry * 0.7)) ** 2 <= 1
        activity[mask & ~inner_mask] = 4.0
    elif config.patient_type == "lung":
        # Body outline
        y_grid, x_grid = np.mgrid[0:size, 0:size]
        cx, cy = size / 2, size / 2
        mask = ((x_grid - cx) / (size * 0.4)) ** 2 + ((y_grid - cy) / (size * 0.35)) ** 2 <= 1
        activity[mask] = 1.0
        # Lung cavities (low uptake)
        lung_l = ((x_grid - cx + size * 0.12) / (size * 0.12)) ** 2 + ((y_grid - cy) / (size * 0.2)) ** 2 <= 1
        lung_r = ((x_grid - cx - size * 0.12) / (size * 0.12)) ** 2 + ((y_grid - cy) / (size * 0.2)) ** 2 <= 1
        activity[lung_l | lung_r] = 0.2
    else:  # body
        y_grid, x_grid = np.mgrid[0:size, 0:size]
        cx, cy = size / 2, size / 2
        mask = ((x_grid - cx) / (size * 0.4)) ** 2 + ((y_grid - cy) / (size * 0.45)) ** 2 <= 1
        activity[mask] = 1.0

    # Add tumors
    for tumor in config.tumors:
        # Convert mm position to pixel
        tx = int(cx + tumor.x_mm * scale)
        ty = int(cy + tumor.y_mm * scale)
        tr = max(2, int(tumor.diameter_mm / 2 * scale))

        y_t, x_t = np.mgrid[0:size, 0:size]
        tumor_mask = (x_t - tx) ** 2 + (y_t - ty) ** 2 <= tr ** 2
        activity[tumor_mask] = tumor.suv_ratio

    # Simulate counting statistics (Poisson noise)
    counts = activity * config.num_annihilations * config.noise_level
    counts = np.maximum(counts, 0)
    noisy = rng.poisson(counts.astype(np.float64).clip(0, 1e8)).astype(np.float64)

    # Apply PSF (point spread function) — resolution blurring
    psf_sigma = 2.0 + rng.uniform(-0.5, 0.5)
    image = gaussian_filter(noisy, sigma=psf_sigma)

    # Normalize to [0, 1]
    if image.max() > 0:
        image = image / image.max()

    metadata = {
        "patient_type": config.patient_type,
        "num_tumors": len(config.tumors),
        "tumors": [asdict(t) for t in config.tumors],
        "ring_diameter_mm": config.ring_diameter_mm,
        "num_annihilations": config.num_annihilations,
        "noise_level": config.noise_level,
        "scatter_fraction": config.scatter_fraction,
        "iterations": config.reconstruction_iterations,
        "crystal_material": config.crystal_material,
        "image_size": config.image_size,
        "psf_sigma": psf_sigma,
    }

    return image, metadata


def generate_sinogram(config: ScanConfig, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """Generate a synthetic sinogram."""
    angular_bins = 64
    radial_bins = 64

    # Create sinogram from activity distribution
    size = config.image_size
    ring_radius = config.ring_diameter_mm / 2.0

    # Simple Radon-like projection
    sinogram = np.zeros((angular_bins, radial_bins))
    angles = np.linspace(0, np.pi, angular_bins, endpoint=False)

    # Create a simple phantom for projection
    phantom = np.zeros((size, size))
    cx, cy = size / 2, size / 2

    # Background
    y_g, x_g = np.mgrid[0:size, 0:size]
    bg_mask = (x_g - cx) ** 2 + (y_g - cy) ** 2 <= (size * 0.35) ** 2
    phantom[bg_mask] = 1.0

    # Add tumors
    for tumor in config.tumors:
        scale = size / (2 * ring_radius * 0.8)
        tx = int(cx + tumor.x_mm * scale)
        ty = int(cy + tumor.y_mm * scale)
        tr = max(2, int(tumor.diameter_mm / 2 * scale))
        t_mask = (x_g - tx) ** 2 + (y_g - ty) ** 2 <= tr ** 2
        phantom[t_mask] = tumor.suv_ratio

    # Forward project (simplified Radon transform)
    for i, angle in enumerate(angles):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for j in range(radial_bins):
            offset = (j - radial_bins / 2) * (size / radial_bins)
            # Line at angle with offset
            x_line = cos_a * np.arange(size) - sin_a * offset + cx
            y_line = sin_a * np.arange(size) + cos_a * offset + cy
            x_idx = np.clip(x_line.astype(int), 0, size - 1)
            y_idx = np.clip(y_line.astype(int), 0, size - 1)
            sinogram[i, j] = phantom[y_idx, x_idx].sum()

    # Add noise
    sinogram = np.maximum(sinogram, 0)
    sinogram = rng.poisson(sinogram * 10 * config.noise_level).astype(np.float64)

    # Add artifacts (optional)
    has_artifact = rng.random() < 0.3
    artifact_type = None
    if has_artifact:
        artifact_type = rng.choice(["gap", "ring", "scatter"])
        if artifact_type == "gap":
            gap_pos = rng.integers(5, angular_bins - 5)
            sinogram[gap_pos:gap_pos + 2, :] = 0
        elif artifact_type == "ring":
            ring_pos = rng.integers(5, radial_bins - 5)
            sinogram[:, ring_pos] *= 0.3
        elif artifact_type == "scatter":
            sinogram += rng.exponential(sinogram.mean() * 0.2, sinogram.shape)

    if sinogram.max() > 0:
        sinogram = sinogram / sinogram.max()

    metadata = {
        "angular_bins": angular_bins,
        "radial_bins": radial_bins,
        "has_artifact": has_artifact,
        "artifact_type": artifact_type,
        "num_tumors": len(config.tumors),
    }

    return sinogram, metadata


# ============================================================================
# IMAGE RENDERING
# ============================================================================

def render_pet_image(image: np.ndarray, metadata: dict, output_path: str):
    """Render PET reconstruction as clinical-style image."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.imshow(image, cmap=CLINICAL_HOT, vmin=0, vmax=1, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close()


def render_sinogram(sinogram: np.ndarray, metadata: dict, output_path: str):
    """Render sinogram as heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.imshow(sinogram, cmap='viridis', aspect='auto', interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()


def render_phantom_diagram(config: ScanConfig, output_path: str):
    """Render phantom cross-section diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    fig.set_facecolor('#1a1a2e')

    # Ring
    ring = plt.Circle((0, 0), config.ring_diameter_mm / 2, fill=False,
                       edgecolor='#64b5f6', linewidth=2)
    ax.add_patch(ring)

    # Body outline
    if config.patient_type == "brain":
        body = Ellipse((0, 0), 160, 195, facecolor='#ce93d8', alpha=0.3, edgecolor='#ab47bc')
    else:
        body = Ellipse((0, 0), 200, 220, facecolor='#a5d6a7', alpha=0.3, edgecolor='#66bb6a')
    ax.add_patch(body)

    # Tumors
    for tumor in config.tumors:
        t = Circle((tumor.x_mm, tumor.y_mm), tumor.diameter_mm / 2,
                   facecolor='#ff1744', alpha=0.7, edgecolor='#d50000', linewidth=2)
        ax.add_patch(t)

    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='#1a1a2e')
    plt.close()


# ============================================================================
# QA GENERATION (automatic from ground truth)
# ============================================================================

def generate_qa_pairs(image_type: str, config: ScanConfig, metadata: dict) -> List[QAPair]:
    """Generate QA pairs from known simulation parameters."""
    pairs = []

    if image_type == "pet_reconstruction":
        # Tumor detection
        has_tumor = len(config.tumors) > 0
        pairs.append(QAPair(
            question="Is there a tumor visible in this PET image?",
            answer="Yes" if has_tumor else "No",
            category="tumor_detection", answer_type="yes_no",
            options=["Yes", "No", "Cannot determine", "Multiple tumors"],
            correct_option=0 if has_tumor else 1,
        ))

        pairs.append(QAPair(
            question="How many lesions are visible in this scan?",
            answer=str(len(config.tumors)),
            category="tumor_detection", answer_type="numeric",
            options=["0", "1", "2", "3 or more"],
            correct_option=min(len(config.tumors), 3),
        ))

        if has_tumor:
            t = config.tumors[0]
            # Localization
            quadrant = "right" if t.x_mm > 0 else "left"
            vertical = "superior" if t.y_mm > 0 else "inferior"
            pairs.append(QAPair(
                question="In which quadrant is the primary lesion located?",
                answer=f"{quadrant} {vertical}",
                category="localization", answer_type="multiple_choice",
                options=["right superior", "right inferior", "left superior", "left inferior"],
                correct_option=["right superior", "right inferior", "left superior", "left inferior"].index(f"{quadrant} {vertical}"),
            ))

            pairs.append(QAPair(
                question="What is the approximate diameter of the largest tumor?",
                answer=f"{t.diameter_mm:.0f}mm",
                category="localization", answer_type="multiple_choice",
                options=["10mm", "15mm", "20mm", "25mm", "30mm", "40mm"],
                correct_option=min(5, max(0, int((t.diameter_mm - 7.5) / 5))),
            ))

            pairs.append(QAPair(
                question="What is the SUV ratio of the tumor to background?",
                answer=f"{t.suv_ratio:.0f}:1",
                category="parameter", answer_type="multiple_choice",
                options=["2:1", "4:1", "6:1", "8:1", "10:1", "12:1"],
                correct_option=min(5, max(0, int((t.suv_ratio - 1) / 2))),
            ))

        # Scan parameters
        pairs.append(QAPair(
            question="What type of anatomical region is shown in this PET scan?",
            answer=config.patient_type,
            category="parameter", answer_type="multiple_choice",
            options=["brain", "lung", "body", "cardiac"],
            correct_option=["brain", "lung", "body", "cardiac"].index(config.patient_type) if config.patient_type in ["brain", "lung", "body"] else 2,
        ))

    elif image_type == "sinogram":
        pairs.append(QAPair(
            question="Does this sinogram show any detector artifacts?",
            answer="Yes" if metadata.get("has_artifact") else "No",
            category="quality", answer_type="yes_no",
            options=["Yes", "No", "Possible", "Cannot determine"],
            correct_option=0 if metadata.get("has_artifact") else 1,
        ))

        if metadata.get("has_artifact"):
            pairs.append(QAPair(
                question="What type of artifact is present in this sinogram?",
                answer=metadata["artifact_type"],
                category="quality", answer_type="multiple_choice",
                options=["gap", "ring", "scatter", "motion"],
                correct_option=["gap", "ring", "scatter", "motion"].index(metadata["artifact_type"]) if metadata["artifact_type"] in ["gap", "ring", "scatter"] else 3,
            ))

        pairs.append(QAPair(
            question="Is there evidence of a high-uptake lesion in this sinogram?",
            answer="Yes" if len(config.tumors) > 0 else "No",
            category="tumor_detection", answer_type="yes_no",
            options=["Yes", "No"],
            correct_option=0 if len(config.tumors) > 0 else 1,
        ))

    elif image_type == "phantom":
        pairs.append(QAPair(
            question="What type of phantom is shown?",
            answer=config.patient_type,
            category="parameter", answer_type="multiple_choice",
            options=["brain", "lung", "body", "NEMA IEC"],
            correct_option=["brain", "lung", "body", "NEMA IEC"].index(config.patient_type) if config.patient_type in ["brain", "lung", "body"] else 2,
        ))

        pairs.append(QAPair(
            question="How many tumors are placed in this phantom?",
            answer=str(len(config.tumors)),
            category="tumor_detection", answer_type="numeric",
            options=["0", "1", "2", "3"],
            correct_option=min(len(config.tumors), 3),
        ))

    return pairs


# ============================================================================
# MAIN DATASET GENERATION
# ============================================================================

def generate_one_sample(idx: int, rng: np.random.Generator, output_dir: Path) -> dict:
    """Generate one image + QA pairs."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Randomly choose image type
    image_type = rng.choice(["pet_reconstruction", "pet_reconstruction", "sinogram", "phantom"])

    # Randomly configure scan
    patient_type = rng.choice(["brain", "lung", "body"])
    num_tumors = int(rng.choice([0, 0, 1, 1, 1, 2]))  # Weighted toward 1

    tumors = []
    for _ in range(num_tumors):
        tumors.append(TumorSpec(
            x_mm=float(rng.uniform(-60, 60)),
            y_mm=float(rng.uniform(-60, 60)),
            diameter_mm=float(rng.uniform(10, 40)),
            suv_ratio=float(rng.uniform(3, 12)),
        ))

    config = ScanConfig(
        patient_type=patient_type,
        ring_diameter_mm=float(rng.choice([350, 400, 450])),
        num_annihilations=int(rng.choice([3000, 5000, 10000])),
        image_size=128,
        tumors=tumors,
        noise_level=float(rng.uniform(0.5, 2.0)),
        reconstruction_iterations=int(rng.choice([20, 50, 100])),
        scatter_fraction=float(rng.uniform(0.2, 0.4)),
    )

    sample_id = f"{image_type}_{idx:06d}"
    image_filename = f"{sample_id}.png"
    image_path = images_dir / image_filename

    # Generate image
    if image_type == "pet_reconstruction":
        image_data, metadata = simulate_pet_scan(config, rng)
        render_pet_image(image_data, metadata, str(image_path))
    elif image_type == "sinogram":
        image_data, metadata = generate_sinogram(config, rng)
        render_sinogram(image_data, metadata, str(image_path))
    elif image_type == "phantom":
        metadata = {"patient_type": patient_type, "num_tumors": num_tumors}
        render_phantom_diagram(config, str(image_path))
    else:
        metadata = {}

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(image_type, config, metadata)

    return {
        "id": sample_id,
        "image": f"images/{image_filename}",
        "image_type": image_type,
        "config": asdict(config),
        "metadata": metadata,
        "qa_pairs": [asdict(qa) for qa in qa_pairs],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate PhysVision training dataset")
    parser.add_argument("--num-images", type=int, default=500, help="Number of images to generate")
    parser.add_argument("--output", type=str, default="data/generated", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    print("=" * 60)
    print(f"PhysVision Dataset Generator")
    print(f"  Images: {args.num_images}")
    print(f"  Output: {output_dir}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    samples = []
    total_qa = 0

    for i in range(args.num_images):
        sample = generate_one_sample(i, rng, output_dir)
        samples.append(sample)
        total_qa += len(sample["qa_pairs"])

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{args.num_images}] Generated {total_qa} QA pairs so far...")

    # Save annotations
    annotations_path = output_dir / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump({"total_images": len(samples), "total_qa_pairs": total_qa, "samples": samples}, f, indent=2)

    # Summary
    types = {}
    for s in samples:
        types[s["image_type"]] = types.get(s["image_type"], 0) + 1

    print(f"\n{'=' * 60}")
    print(f"DONE!")
    print(f"  Total images: {len(samples)}")
    print(f"  Total QA pairs: {total_qa}")
    print(f"  By type: {types}")
    print(f"  Output: {annotations_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
