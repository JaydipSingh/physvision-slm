# PhysVision-SLM: Simulation-to-Model Multimodal SLM for Medical Physics

## Paper Title (Working)

**"Synthetic-to-Model: Training a Multimodal Small Language Model for Medical Physics Image Understanding Using Simulation-Generated Data"**

---

## 1. Executive Summary

We present a novel pipeline that combines a **Geant4-based medical imaging simulator** (g4_agent) with a **multimodal small language model** (PhysVision-SLM) to create an AI system that can understand and reason about medical physics images — trained entirely on synthetic data with automatic ground-truth annotations.

**Key Innovation**: The simulator generates unlimited labeled training data (PET images, sinograms, phantom diagrams + QA pairs) without any human annotation cost. The SLM learns to interpret these images and answer questions about them.

**Two language backbones compared**:
1. **TinyLMv3 (35M)** — custom, continues paper 1's constrained-hardware story
2. **Qwen2-0.5B** — pre-trained, production-level quality

**Hardware**: Apple M3 Pro (18GB) — same as paper 1.

---

## 2. Research Questions

1. Can a physics simulator serve as a zero-cost labeled data generator for training multimodal models?
2. How does simulation-trained visual understanding transfer to realistic medical imaging scenarios?
3. Can a <500M parameter model perform meaningful medical physics visual reasoning?
4. What is the minimum dataset size from simulation needed for effective visual QA?
5. Which medical physics visual tasks benefit most from simulation-generated training?

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     SIMULATION-TO-MODEL PIPELINE                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────────────────────────────────┐                      │
│  │          g4_agent (Data Generator)           │                      │
│  │                                              │                      │
│  │  PETGeometryAgent → Ring/Block Detectors    │                      │
│  │  PhantomAgent → Brain/Lung/NEMA Phantoms    │                      │
│  │  PETSimulationAgent → Coincidence Events    │                      │
│  │  ReconstructionAgent → MLEM/OSEM Images     │                      │
│  │                                              │                      │
│  │  OUTPUT: Images + Ground-Truth Annotations   │                      │
│  └──────────────────┬───────────────────────────┘                      │
│                     │                                                   │
│                     ▼                                                   │
│  ┌─────────────────────────────────────────────┐                      │
│  │       Auto-Generated Training Dataset        │                      │
│  │                                              │                      │
│  │  50K+ image-question-answer triplets:        │                      │
│  │  • "Where is the tumor?" → (30,20) mm       │                      │
│  │  • "What is the SUV ratio?" → 8:1           │                      │
│  │  • "Is this sinogram normal?" → No, scatter │                      │
│  │  • "How many rings?" → 4                    │                      │
│  │  • "What reconstruction was used?" → MLEM   │                      │
│  └──────────────────┬───────────────────────────┘                      │
│                     │                                                   │
│                     ▼                                                   │
│  ┌─────────────────────────────────────────────┐                      │
│  │          PhysVision-SLM (Learner)            │                      │
│  │                                              │                      │
│  │  ┌────────┐  ┌──────────┐  ┌────────────┐  │                      │
│  │  │ SigLIP │→ │Projection│→ │ Language   │  │                      │
│  │  │(frozen)│  │  (MLP)   │  │   Model    │  │                      │
│  │  └────────┘  └──────────┘  │            │  │                      │
│  │                             │ TinyLMv3   │  │                      │
│  │                             │   OR       │  │                      │
│  │                             │ Qwen2-0.5B │  │                      │
│  │                             └────────────┘  │                      │
│  └─────────────────────────────────────────────┘                      │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Roles

| Component | Source | Role |
|-----------|--------|------|
| g4_agent PETGeometryAgent | Existing code | Generate detector geometry images |
| g4_agent PhantomAgent | Existing code | Generate phantom cross-sections with tumors |
| g4_agent PETSimulationAgent | Existing code | Generate sinograms and hit data |
| g4_agent ReconstructionAgent | Existing code | Generate reconstructed PET images |
| **NEW** DatasetGenerator | To build | Orchestrate g4_agent → produce image+QA pairs |
| **NEW** PhysVision model | To build | SigLIP + projection + LM |
| **NEW** Eval pipeline | To build | Benchmark on held-out + external data |

---

## 4. Data Generation Strategy

### 4.1 Why Simulation Data Is Superior

**Note on Physics Simulation**: We use analytical simulation (ray-cylinder intersection, Poisson statistics, Gaussian PSF) rather than full Geant4 Monte Carlo. This produces visually realistic images with exact ground-truth annotations at 1000× the speed of Monte Carlo. For training a visual QA model, image-level realism and correct labels matter more than exact particle transport physics. A Geant4 validation study comparing analytical vs Monte Carlo images is planned as future work.

| Aspect | Manual Labeling | Simulation (g4_agent) |
|--------|----------------|----------------------|
| Cost | $$$$ (radiologist time) | $0 (compute only) |
| Scale | Hundreds (limited budget) | 50,000+ (unlimited) |
| Accuracy | Human error, inter-rater variability | Exact ground truth |
| Diversity | Limited by available cases | Controllable variation |
| Privacy | HIPAA/patient consent issues | No privacy concerns |
| Reproducibility | Hard to replicate | Deterministic (seed-based) |

### 4.2 Image Types from g4_agent

| Type | Generator | Parameters Varied | Example QA |
|------|-----------|-------------------|-----------|
| **PET reconstructions** | PETReconstructionAgent | tumor size, position, SUV, noise | "Where is the lesion?" |
| **Sinograms** | PETSimulationAgent | event count, scatter fraction, gaps | "Is there a detector gap artifact?" |
| **Phantom diagrams** | PhantomAgent | anatomy type, lesion count, uptake | "How many tumors are present?" |
| **Detector geometry** | PETGeometryAgent | ring count, crystal size, material | "What crystal material is used?" |
| **Quality control** | Varied parameters | resolution, uniformity, contrast | "Does this pass QC?" |
| **Noisy vs clean** | Add Poisson noise | noise level, iteration count | "Is this image under-iterated?" |

### 4.3 Question-Answer Generation (Automatic)

Since we control the simulation parameters, we auto-generate QA pairs:

```python
# Example: generating one image + QA triplet
phantom_config = PhantomConfig(
    patient_type="brain",
    lesions=[LesionSpec(position_mm=(30, 20, 40), diameter_mm=25, uptake_ratio=8.0)]
)
phantom, uptake_map = phantom_agent.create_phantom(phantom_config)
pet_image = reconstruct(simulate(phantom))

# Auto-generated questions from known ground truth:
qa_pairs = [
    {"question": "Is there a tumor in this image?", "answer": "Yes"},
    {"question": "Where is the tumor located?", "answer": "Right frontal region, (30,20)mm from center"},
    {"question": "What is the tumor diameter?", "answer": "25mm"},
    {"question": "What is the SUV ratio?", "answer": "8:1 (tumor:background)"},
    {"question": "What type of scan is this?", "answer": "FDG-PET brain scan"},
    {"question": "How many lesions are visible?", "answer": "1"},
]
```

### 4.4 Dataset Scale Plan

| Phase | Images | QA Pairs | Time to Generate |
|-------|--------|----------|-----------------|
| Pilot (testing) | 500 | 3,000 | ~30 min |
| Training | 10,000 | 50,000 | ~5 hours |
| Full | 20,000 | 100,000 | ~10 hours |

Each image generates 5-10 QA pairs (multi-turn potential).

---

## 5. Model Architecture Details

### 5.1 Vision Encoder

**SigLIP-SO400M/14** (frozen)
- Input: 224×224 or 384×384 image (PET scans, sinograms)
- Output: 729 patch tokens × 1152 dimensions
- Why SigLIP: Better than CLIP for scientific/medical images, trained on larger dataset

### 5.2 Projection Layer

```python
class VisionProjection(nn.Module):
    """Projects vision features into language model embedding space."""
    def __init__(self, vision_dim=1152, text_dim=384, num_tokens=64):
        self.compressor = nn.Linear(vision_dim, text_dim)  # Reduce dimension
        self.pooler = nn.AdaptiveAvgPool1d(num_tokens)     # Reduce sequence length
        self.norm = nn.LayerNorm(text_dim)
```

Reduces 729×1152 → 64×384 (manageable for small LM).

### 5.3 Language Backbone A: TinyLMv3 (35M)

From paper 1, custom transformer:
- d_model=384, n_layers=6, n_heads=6
- RoPE, RMSNorm, SwiGLU
- LoRA fine-tuning (r=16 for multimodal)

### 5.4 Language Backbone B: Qwen2-0.5B

Pre-trained:
- 500M params, strong reasoning
- LoRA (r=16, targets: q/k/v/o/gate/up/down)
- Expected significantly better performance

---

## 6. Training Strategy

### Stage 1: Vision-Language Alignment (2 hours)

- **Train**: Projection layer ONLY
- **Freeze**: SigLIP + Language Model
- **Data**: 10K image-caption pairs (g4_agent image + description)
- **Objective**: MSE alignment of vision features to text embedding space

### Stage 2: Instruction Tuning (4-6 hours)

- **Train**: Projection + LoRA on language model
- **Freeze**: SigLIP
- **Data**: 50K image+question→answer triplets
- **Objective**: Cross-entropy on answer tokens (standard VQA training)

### Stage 3: Multi-Turn Dialog (Optional, 2 hours)

- **Train**: LoRA only (projection frozen)
- **Data**: Multi-turn conversations about images
- **Objective**: Follow-up question handling

---

## 7. Evaluation Plan

### 7.1 Internal Benchmarks (from g4_agent)

| Benchmark | Questions | Image Type | Metric |
|-----------|-----------|-----------|--------|
| **Tumor Detection QA** | 500 | PET reconstructions | Accuracy (yes/no) |
| **Tumor Localization** | 300 | PET with annotations | Position error (mm) |
| **Sinogram Quality** | 200 | Sinograms (clean/artifact) | Accuracy |
| **Detector Parameter QA** | 200 | Geometry diagrams | Exact match |
| **Reconstruction Assessment** | 300 | MLEM vs OSEM comparisons | Accuracy |
| **Multi-choice Physics** | 500 | Mixed | Accuracy (4-option) |
| **Total** | **2,000** | — | — |

### 7.2 External Benchmarks

| Benchmark | What | Notes |
|-----------|------|-------|
| ScienceQA (image subset) | General science visual QA | Cross-domain transfer |
| Medical VQA (VQA-RAD) | Radiology visual QA | Clinical transfer |
| PathVQA | Pathology visual QA | Medical domain |

### 7.3 Baselines

| Model | Type | Purpose |
|-------|------|---------|
| Random | — | Lower bound (25% for MCQ) |
| Text-only (no image) | Ablation | How much does vision help? |
| SigLIP zero-shot | Vision only | How much does the LM add? |
| SmolVLM-256M | External small VLM | Competitive baseline |
| GPT-4V (via API) | Large VLM | Upper bound reference |
| Radiologist (human) | Human expert | Clinical reference |

### 7.4 Metrics

- **Accuracy** (MCQ, yes/no)
- **Localization error** (mm, for spatial questions)
- **BLEU/ROUGE** (for open-ended descriptions)
- **Inference latency** (ms on M3 Pro)
- **Memory footprint** (GB)
- **Per-category breakdown** (tumor detection vs quality vs parameters)

---

## 8. Implementation Plan

### Phase 1: Dataset Generator (Week 1)

- [ ] Build `DatasetGenerator` class that wraps g4_agent
- [ ] Implement parameter randomization (tumor size/pos, noise, detector config)
- [ ] Implement automatic QA generation from known parameters
- [ ] Generate pilot dataset (500 images, 3K QA pairs)
- [ ] Validate image quality and QA correctness
- [ ] Save in standard format: `{image_path, question, answer, metadata}`

### Phase 2: Model Architecture (Week 1-2)

- [ ] Implement SigLIP feature extractor wrapper
- [ ] Implement VisionProjection module
- [ ] Integrate with TinyLMv3 (multimodal forward pass)
- [ ] Integrate with Qwen2-0.5B (multimodal forward pass)
- [ ] Implement multimodal training loop

### Phase 3: Data Generation at Scale (Week 2)

- [ ] Generate full training set: 10K images, 50K QA pairs
- [ ] Generate held-out evaluation set: 2K questions
- [ ] Verify dataset balance across categories
- [ ] Create train/val/test splits

### Phase 4: Training (Week 2-3)

- [ ] Stage 1: Alignment pre-training (2 hours)
- [ ] Stage 2: Instruction tuning (4-6 hours)
- [ ] Compare TinyLMv3 vs Qwen2-0.5B
- [ ] Monitor generation health (no EOS collapse)
- [ ] Checkpoint selection by validation accuracy

### Phase 5: Evaluation (Week 3-4)

- [ ] Run all internal benchmarks (2000 questions)
- [ ] Run external benchmarks (ScienceQA, VQA-RAD if accessible)
- [ ] Run baselines (GPT-4V via API, SmolVLM)
- [ ] Per-category analysis
- [ ] Ablation studies (vision contribution, dataset size scaling)

### Phase 6: Paper Writing (Week 4-5)

- [ ] Write paper (strong narrative: simulation → model)
- [ ] Generate figures (pipeline, accuracy curves, example predictions)
- [ ] Submit to arXiv
- [ ] Target venue: MICCAI Workshop, NeurIPS ML4Health, or EMNLP

---

## 9. Expected Results

| Model | Tumor QA | Sinogram QA | Overall | Latency |
|-------|----------|-------------|---------|---------|
| Random | 50% | 50% | 25% | — |
| Text-only | 55% | 40% | 35% | 50ms |
| **PhysVision-TinyLM** | 75% | 65% | 55-60% | 100ms |
| **PhysVision-Qwen** | 90% | 80% | 75-80% | 200ms |
| GPT-4V | 95% | 90% | 90% | 2000ms |

---

## 10. Novelty & Contribution

1. **Simulation-to-Model pipeline** — First to use a physics simulator as zero-cost labeled data generator for multimodal SLM training (no human annotation)
2. **Medical physics visual QA benchmark** — New benchmark with 2000 questions + ground-truth from simulation
3. **Domain-specialized multimodal SLM** — Under 1B params, runs on laptop, medical physics specific
4. **Dataset scaling analysis** — How much simulation data is needed for effective visual understanding?
5. **Synthetic-to-real transfer study** — Does simulation-trained model generalize to real clinical images?
6. **Builds on paper 1** — Same hardware, same philosophy (reproducible, constrained), extended to multimodal

---

## 11. Connection to Paper 1 (slm_v0)

| Paper 1 (slm_v0) | Paper 2 (PhysVision) |
|-------------------|---------------------|
| Text-only SLM | Multimodal (vision + text) SLM |
| Rust BPE tokenizer | Same tokenizer (for text component) |
| arXiv physics corpus | g4_agent simulation data |
| 20-question QA | 2000-question visual QA |
| LoRA on FFN only | LoRA on full attention + FFN |
| EOS collapse documented | EOS collapse fixed (doc packing from v3) |
| GPT-2 baseline | GPT-4V + SmolVLM baselines |
| Training pipeline | Same pipeline + vision encoder |

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| SigLIP doesn't understand medical images well | Fine-tune last 2 layers; or use BiomedCLIP |
| Simulation images look too synthetic | Add realistic noise, PSF blur, patient motion |
| TinyLMv3 (35M) too small for visual reasoning | Qwen2-0.5B as primary; TinyLM as ablation |
| Synthetic→real domain gap | Include a small set of real clinical-style images in eval |
| Training EOS collapse | Already fixed in v3 with document packing |
| g4_agent generation too slow | Batch generation overnight; cache images |

---

## 13. File Structure

```
physvision_slm/
├── spec/
│   └── PROJECT_SPEC.md              (this file)
├── src/
│   ├── dataset_generator.py         (g4_agent → image+QA pairs)
│   ├── vision_encoder.py            (SigLIP wrapper)
│   ├── projection.py                (MLP bridge)
│   ├── physvision_model.py          (full multimodal model)
│   └── train_multimodal.py          (training loop)
├── scripts/
│   ├── generate_dataset.py          (run g4_agent to produce data)
│   ├── eval_internal.py             (our benchmark)
│   ├── eval_external.py             (ScienceQA, VQA-RAD)
│   └── compare_baselines.py         (GPT-4V, SmolVLM)
├── data/
│   ├── generated/                   (simulation output)
│   │   ├── images/                  (PNG files)
│   │   └── annotations.json         (QA pairs + metadata)
│   └── benchmarks/                  (eval datasets)
├── doc/
│   └── paper_draft.tex
├── config/
│   └── training_config.yaml
├── g4_agent/                        (symlink or copy of g4_agent code)
└── README.md
```

---

## 14. Timeline

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Dataset generator + pilot data | 500 images, 3K QA pairs, working pipeline |
| 2 | Full dataset + model architecture | 10K images, 50K QA, model code |
| 3 | Training complete | Both backbones trained, initial eval |
| 4 | Full evaluation + baselines | All benchmarks, comparison tables |
| 5 | Paper draft + submission | arXiv upload |

**Total: 5 weeks to submission**

---

## 15. Target Venues

| Venue | Why | Deadline |
|-------|-----|----------|
| **MICCAI 2027 Workshop** | Medical imaging + AI, perfect fit | ~Mar 2027 |
| **NeurIPS ML4Health** | Health ML workshop | ~Sep 2027 |
| **EMNLP 2027** | NLP + multimodal, industry track | ~Jun 2027 |
| **Medical Image Analysis (journal)** | Top medical imaging journal | Rolling |
| **arXiv** (immediate) | Timestamp the work | Anytime |

---

## 16. Key References

- Our Paper 1: "A Reproducible Systems Stack for Domain-Specialized SLMs" (2026)
- SmolVLM: "Redefining small and efficient multimodal models" (arXiv:2504.05299)
- TinyLLaVA: "A Framework of Small-scale Large Multimodal Models" (arXiv:2402.14289)
- TinyAlign: "Retrieval-augmented lightweight VLM" (ACL Findings 2026)
- "Crafting Effective Small Language Models and Multimodal SLMs" (arXiv:2502.11573)
- "Multimodal LLMs and Physics Visual Tasks" (IOP 2025)
- Multi-Physics Benchmark (arXiv:2509.15839)
- ScienceQA (2022)
- VQA-RAD: "Visual Question Answering in Radiology"
