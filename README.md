# ViT-AD-Opt: Vision Transformer Optimization Suite for Anomaly Detection

**Course Project — Embedded AI**  
**Submitted to:** **DR. DUBACHARLA GYANESHWAR**  

---

## Team Members & Info

| Name | Roll Number | Department / Major |
| :--- | :--- | :--- |
| **Abhishek** (Team Lead) | `AD23B1012` | Artificial Intelligence & Data Science |
| **Lalith Karthik** | `CS23B1042` | Computer Science & Engineering |
| **Jashwanth G** | `AD23B1020` | Artificial Intelligence & Data Science |
| **N Murthy** | `AD23B1034` | Artificial Intelligence & Data Science |

---

## Project Overview

This repository contains **ViT-AD-Opt**, an end-to-end model optimization suite designed for **Vision Transformer (ViT)** architectures applied to **Anomaly Detection** in video surveillance. 

Using the **ShanghaiTech (SHTech)** video anomaly detection dataset, we construct a reconstruction-based anomaly detection framework. Given a video frame, the model extracts patches, processes them through a pre-trained Vision Transformer backbone (`ViT-B/16` with approximately 86 million parameters and 12 transformer encoder layers), and attempts to reconstruct the original pixel values of the patches using a custom `PatchReconstructionHead`. Abnormal behaviors (e.g., fast movement, unusual objects) result in higher reconstruction errors, which serve as the anomaly score.

Because deploying large Vision Transformers on resource-constrained embedded and edge devices is challenging due to high latency, memory, and power constraints, this project focuses on **four critical paradigms of model compression and optimization**:

1. **Model Pruning (Sparsification)**
2. **Model Quantization (Low Precision)**
3. **Knowledge Distillation (KD)**
4. **Token Reduction (ToMe & DynamicViT)**

---

## Technical Architecture & Methodology

The core code is consolidated into `notebookaf675a5a93.ipynb` which guides the user through data extraction, model training, multiple optimization experiments, latency/memory profiling, and visualization.

```
                  ┌──────────────────────────────────────────────┐
                  │          Input Video Frame (160x160)         │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │        Patch Extraction (10x10 Patches)       │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Vision Transformer (ViT/DeiT Backbone)    │
                  └──────────────────────┬───────────────────────┘
                                         │  ◄── Pruned / Quantized /
                                         │      Token Merged Backbone
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │         Patch Reconstruction Head            │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │       Mean Squared Error (MSE) Loss          │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │       Anomaly Score & Frame-Level ROC-AUC    │
                  └──────────────────────────────────────────────┘
```

### 1. Model Pruning (Sparsification)
Pruning eliminates redundant parameters to speed up inference and reduce model size. We implement and evaluate four strategies:
* **Unstructured $L_1$ One-Shot Pruning:** Prunes individual weights across all linear layers based on absolute magnitude.
* **Iterative Pruning:** Prunes weights gradually over multiple cycles with fine-tuning epochs to recover lost accuracy.
* **Taylor-Based Head Pruning:** Measures the gradient sensitivity of individual attention heads and prunes the least important heads.
* **FFN Channel Pruning:** Targets the MLP feed-forward expansion layers, pruning entire channels to reduce matrix dimensions.

### 2. Model Quantization (Low Precision)
Quantization maps 32-bit floating-point weights and activations to lower bit-widths, drastically reducing model size and enabling hardware acceleration:
* **FP16 Mixed Precision:** Standard float16 wrapper for native GPU speedup.
* **Dynamic Post-Training Quantization (PTQ) INT8:** Quantizes weights to 8-bit integers dynamically during runtime.
* **Static PTQ INT8:** Calibrates activation distributions using validation batches to compute static scale and zero-point parameters.
* **Quantization-Aware Training (QAT) INT8:** Simulates quantization noise in the forward pass during training, allowing the model to adapt to integer restrictions and maintain accuracy.
* **BitsAndBytes INT4 (NF4):** Employs NormalFloat4 quantization to compress the backbone into 4 bits, ideal for extreme edge constraints.

### 3. Knowledge Distillation (KD)
We compress a large teacher model (`ViT-B/16`) into a highly efficient student model (`DeiT-Tiny`). We design a custom multi-component loss function (`KDLoss`):
$$\mathcal{L}_{KD} = w_{recon}\mathcal{L}_{MSE}(S_{recon}, T_{recon}) + w_{logit}\mathcal{L}_{logit} + w_{feat}\mathcal{L}_{feat} + w_{attn}\mathcal{L}_{attn}$$
This forces the student to mimic the reconstruction capabilities, final representations, intermediate layer features, and self-attention patterns of the teacher.

### 4. Token Reduction
Vision Transformers process a fixed number of tokens. We accelerate inference by reducing the token count dynamically inside the backbone:
* **Token Merging (ToMe):** Bipartite matching merges similar patch tokens dynamically without re-training.
* **DynamicViT:** Integrates a learned lightweight `TokenPredictor` network that scores tokens and drops the least relevant ones progressively through the transformer layers.

---

## Required Dependencies & Libraries

To run this project, make sure you have a Python environment with PyTorch and CUDA support. You can install all necessary packages by running:

```bash
pip install -q transformers==4.41.0 \
                 timm==0.9.16 \
                 fvcore \
                 thop \
                 codecarbon \
                 torchmetrics \
                 bitsandbytes \
                 opencv-python-headless \
                 rich \
                 pandas \
                 numpy \
                 scikit-learn \
                 matplotlib \
                 seaborn \
                 tqdm
```

---

## Steps to Run the Code

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ad23b1012/EAI-TEAM-1.git
   cd EAI-TEAM-1
   ```

2. **Acquire the ShanghaiTech Dataset:**
   The notebook is configured to auto-detect Kaggle's input directories. If running locally, place the ShanghaiTech dataset in your data directory:
   * Training videos: `<DATA_DIR>/training/videos/`
   * Testing frames: `<DATA_DIR>/testing/frames/`
   * Anomaly masks: `<DATA_DIR>/testing/test_frame_mask/`

3. **Open and Run the Jupyter Notebook:**
   Launch Jupyter Lab or Notebook:
   ```bash
   jupyter notebook notebookaf675a5a93.ipynb
   ```
   * Execute **CELL 1 to CELL 7** to install dependencies, set up configuration paths, and pre-extract frames from training videos.
   * Execute **CELL 8 to CELL 13** to build the reconstruction architecture, configure caching helpers, and set up checkpoint directories.
   * Execute **CELL 14 to CELL 18** to train the baseline model (`ViT-B/16`) and log its baseline ROC-AUC, latency, and memory metrics.
   * Execute **CELL 19 to CELL 38** sequentially to run all optimization experiments (Pruning, Quantization, Distillation, Token Reduction) and generate comparative plots.

---

## Expected Outputs & Verification

Upon running all experiments, the notebook automatically:
1. Logs execution statistics, hardware parameters, carbon footprints (using `codecarbon`), and profiling times.
2. Generates four distinct comparison plots:
   * **Pruning Trade-offs:** ROC-AUC vs. Sparsity / Latency under different pruning strategies.
   * **Quantization Benchmarks:** Model Size (MB) and Latency (ms) comparison across FP32, FP16, PTQ INT8, QAT INT8, and NF4 INT4.
   * **Distillation Improvement:** ROC-AUC curves comparing Student-Only, Teacher-Only, and Distilled Student models.
   * **Token Reduction Profile:** Inference speed (tokens/sec) vs. Reconstruction error for varying ToMe merging ratios and DynamicViT keep ratios.
3. Prints a **Final Summary Table** listing each model's:
   * Model Name & Applied Optimization Method
   * Total Parameters (Millions)
   * Disk Size (MB)
   * Average Latency per Frame (ms) on CPU/GPU
   * Peak VRAM / Host RAM Usage (MB)
   * Frame-Level Anomaly Detection ROC-AUC score

This summary table serves as the primary verification artifact to prove the correctness and real-world efficiency gains of the optimization suite.

---

## Submission Package Structure (ZIP/RAR)

For university submission as instructed by the professor, the compressed archive should be prepared as follows:

```
EAI-TEAM-1.zip
├── notebookaf675a5a93.ipynb         # Source code & experiment notebook
├── README.md                        # Documentation (this file)
└── individual_contributions.pdf     # Single PDF containing signed and dated handwritten notes (≤100 words per member)
```

### Individual Contributions Breakdown 
* **Abhishek (AD23B1012):** Contributed to improving embedded deployment efficiency through model compression techniques. Work involved implementing quantization methods such as FP16 and INT8, as well as knowledge distillation from the ViT-B/16 teacher model to a DeiT-Tiny student model. Continuously evaluated model size, inference latency, and accuracy trade-offs, while also assisting in debugging and analyzing optimized configurations for embedded anomaly detection.
* **Lalith Karthik (CS23B1042):** Contributed to implementing optimization techniques aimed at improving Vision Transformer efficiency. Work included experimenting with pruning strategies such as L1 pruning, iterative pruning, attention head pruning, and FFN channel pruning. Also explored literature-based methods related to transformer attention optimization and token efficiency, continuously testing different configurations and evaluating their effect on anomaly detection performance and computational cost.
* **Jashwanth G (AD23B1020):** Contributed to the initial research and baseline development of the project. Work involved surveying literature related to efficient Vision Transformers, including methods such as Swin Transformers, DeiT, token reduction, and attention optimization techniques suitable for embedded deployment. Also worked on implementing and benchmarking the initial ViT-B/16 baseline model, obtaining reference metrics, and assisting in debugging the training and evaluation pipeline throughout the project.
* **N Murthy (AD23B1034):** Contributed to experimentation, benchmarking, and integration throughout the project. Work involved validating baseline and optimized model performance using metrics such as AUC-ROC, latency, and model size. Worked on generating comparative results, debugging experiment outputs, and integrating different optimization methods into a unified evaluation pipeline. Continuously assisted in consolidating findings and maintaining consistency across project experiments.
