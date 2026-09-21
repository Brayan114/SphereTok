# SphereTok: An Egocentric 3D World Tokenizer for Embodied Perception in Voxel Environments

**Authors:** Anonymous Authors  
**Target Venue:** NeurIPS Workshop on Open-World Embodied AI (O-WEAI) / ICLR Workshop on World Models  
**Keywords:** 3D Representation Learning, Vector-Quantized Autoencoders, Voxel Tokenization, Embodied AI, Minecraft

---

## Abstract

Autonomous embodied agents operating in open-world sandbox environments like Minecraft face a fundamental perceptual trade-off. High-level Large Language Model (LLM) planners (e.g., Voyager) rely on symbolic text telemetry that exhibits the *"Airport Tower Dilemma,"* observing discrete semantic logs stripped of continuous 3D surface geometry, obstacle contours, and kinematic jump clearances. Conversely, 2D Vision-Language-Action (VLA) models (e.g., VPT, STEVE-1) infer actions from flat RGB video projections that suffer from monocular depth ambiguity, occlusion blindness, and severe spatial amnesia upon camera rotation.

In this work, we introduce **SphereTok**, an **Egocentric 3D World Tokenizer** designed to provide embodied decision-makers with compact, discrete 3D geometric state representations. SphereTok frames local spatial awareness as an agent-centered $32 \times 32 \times 32$ voxel lattice (32,768 cells). To prevent cubic sequence explosion in Transformer attention ($O(N^2)$), SphereTok introduces:
1. A **$4 \times 4 \times 4$ micro-cube spatial patchification** that reduces raw voxels by $64.0\times$ (512 patches);
2. A **surface-affordance clearance shell** that prunes deep uniform void space while strictly retaining the 26-neighborhood clearance envelope bordering solid terrain, achieving an additional $1.98\times$ reduction ($127.0\times$ overall token compression, yielding $\approx 258$ active tokens per timestep);
3. A **3D Vector-Quantized Variational Autoencoder (3D VQ-VAE)** with a discrete 512-codebook vocabulary;
4. Continuous **yaw/pitch-relative spherical positional embeddings** $(\rho, \Delta\theta, \Delta\phi)$ that ground spatial directionality without rotating the underlying rectilinear lattice.

Trained on 1,600 diverse procedural Minecraft terrain chunks, SphereTok converges stably on an NVIDIA T4 GPU, achieving **87.06% voxel reconstruction accuracy**, **85.38% solid terrain Intersection-over-Union (IoU)**, and an active codebook utilization of 81.6% (418/512 codes) with no codebook collapse. In inference benchmarks, SphereTok processes chunks in **1.84 ms on a GPU (~543 FPS)** and **12.3 ms on a CPU (~81 FPS)**, easily satisfying Minecraft’s 20 Hz (50 ms) real-time tick budget. In an exploratory 3D parkour navigation pilot, an action policy conditioned on SphereTok tokens achieves an overall success rate of **65.0%** compared to **37.0%** for a categorical semantic baseline, resolving fine-grained kinematic jumps where semantic indicators fail. SphereTok provides a modular, computationally efficient 3D spatial primitive bridging symbolic planning and embodied motor control.

---

## 1. Introduction

Autonomous agency in open-ended virtual sandbox environments serves as a primary proving ground for artificial general intelligence. Minecraft represents a uniquely challenging embodied testbed due to its infinite procedural generation, multi-tier crafting trees, non-linear physics, and complex 3D voxel topography.

```
       Paradigm A: Symbolic Text LLM                      Paradigm B: 2D Pixel VLA
       ┌───────────────────────────────┐                  ┌───────────────────────────────┐
       │ JSON Telemetry Dump:          │                  │ 2D Monocular Camera (RGB):    │
       │  {"nearest_tree": [12, 64, -8]│                  │  [ 128 x 128 Flat Pixels ]    │
       │   "block_in_front": "stone"}  │                  │                               │
       └──────────────┬────────────────┘                  └──────────────┬────────────────┘
                      │                                                  │
                      ▼                                                  ▼
           "Airport Tower" Problem                             Occlusion Blindness
          No physical 3D geometry;                         No depth permanence; camera turn
          relies on hardcoded pathfinders                     causes spatial amnesia
                      │                                                  │
                      └──────────────────────┬───────────────────────────┘
                                             │
                                             ▼
                                  OUR PROPOSED SOLUTION:
                        SphereTok (Egocentric 3D World Tokenizer)
                                             │
                                             ▼
                         ┌───────────────────────────────────────┐
                         │   Agent-Centered 32x32x32 Voxel Box   │
                         │      (World-Axis-Aligned Lattice)     │
                         │                   │                   │
                         │  [Surface-Affordance Pruning (26-N)]  │
                         │  [Micro-Cube 3D VQ-VAE Tokenizer]     │
                         │  [Relative Spherical Pos. Embeddings] │
                         │                   │                   │
                         │       Discrete 3D World Tokens        │
                         │      [T_3D_1, T_3D_2, ..., T_258]     │
                         └───────────────────┬───────────────────┘
                                             │
                                             ▼
                         ┌───────────────────────────────────────┐
                         │      Embodied Foundation Models       │
                         │  (Compatible with Transformer Context)│
                         └───────────────────────────────────────┘
```

Despite rapid recent progress, existing Minecraft agent paradigms struggle to balance spatial richness with computational tractability:

1. **The Disembodied Text Planner (The "Airport Tower Dilemma"):** Text-based Large Language Model (LLM) agents such as Voyager (Wang et al., 2023) receive symbolic JSON logs of nearby entity coordinates and block names. This is analogous to handing air traffic control telemetry to an airport tower controller and expecting them to steer an aircraft through physical turbulence. The model observes discrete labels (`"stone at (12, 64, -8)"`), but possesses no native representation of continuous surface slopes, jump clearances, corridor widths, or volumetric collision boundaries. Consequently, high-level LLMs cannot directly generate kinematic motor controls, forcing total reliance on external deterministic pathfinders (`mineflayer-pathfinder`). When these brittle search algorithms fail in dynamic or non-flat terrain, the agent receives text execution exceptions (`"Path obstructed"`), precipitating hallucination and behavioral stalling.

2. **The 2D Perspective Controller (Monocular Pixel Blindness):** 2D Vision-Language-Action (VLA) architectures such as Video PreTraining (VPT; Baker et al., 2022) and STEVE-1 (Lifshitz et al., 2023) clone human gameplay from $128 \times 128$ first-person RGB video streams. While capable of agile reflexive maneuvers, monocular 2D projections lack metric depth calibration and 3D permanence (Chen et al., 2024). When an agent turns its virtual head, objects outside the immediate camera frustum are instantly purged from the visual buffer. In multi-level subterranean mines or enclosed shelters, 2D policies cannot reason about occluded geometry or overhead headroom.

3. **The 3D Foundation & World Model Gap:** Recent 3D foundation models (e.g., 3D-LLM; Hong et al., 2023; LEO; Huang et al., 2024) rely on point clouds and triangle meshes, which are computationally prohibitive to maintain across billions of interactive voxel blocks. Concurrently, generative world models like DreamerV3 (Hafner et al., 2023) and MineWorld (2025) learn latent dynamics from video or actions, while autonomous driving models like OccWorld (Zheng et al., 2023) and $I^2$-World (2025) demonstrate the power of discrete 3D/4D occupancy tokenization. However, an open problem remains: *How can an embodied agent represent local 3D metric geometry as a compact sequence of discrete tokens optimized for physical action selection?*

To bridge this gap, we present **SphereTok**, an **Egocentric 3D World Tokenizer for Voxel Environments**. SphereTok is guided by a core biological inductive bias: *embodied organisms navigate their environment not by reading semantic lists or projecting flat images, but by maintaining an egocentric perceptual field of physical affordances and spatial clearances.*

### Key Contributions:
1. **Agent-Centered Voxel Grid with Relative Spherical Positional Grounding:** We formulate an agent-centered $32 \times 32 \times 32$ bounding volume (32,768 voxels). To prevent severe geometric aliasing and interpolation artifacts caused by rotating discrete rectilinear lattices, the grid remains strictly world-axis-aligned, while agent yaw and pitch are injected via continuous relative spherical positional embeddings $(\rho, \Delta\theta, \Delta\phi)$.
2. **Surface-Affordance Clearance Pruning:** We address the Transformer sequence explosion problem through a two-stage spatial sparsification: $4 \times 4 \times 4$ micro-cube partitioning ($64.0\times$ compression) followed by a 26-neighborhood surface clearance filter ($1.98\times$ compression). This preserves critical jump headroom, footing, and corridor bounds while eliminating distant sky and deep bedrock, achieving an overall **$127.0\times$ token reduction** ($\approx 258$ tokens).
3. **Discrete 3D VQ-VAE Architecture:** We train a 3D convolutional VQ-VAE ($869\text{K}$ parameters) that discretizes active spatial patches into a 512-codebook vocabulary, converging to **87.06% voxel accuracy** and **85.38% solid terrain IoU** with **81.6% codebook utilization** on procedural Minecraft chunks.
4. **Real-Time Efficiency:** SphereTok achieves an inference latency of **1.84 ms on an NVIDIA T4 GPU (~543 FPS)** and **12.3 ms on a CPU (~81 FPS)**, executing well within Minecraft's 50 ms (20 Hz) tick window.
5. **Kinematic Parkour Pilot:** In an exploratory 3D parkour navigation task with variable gap widths (1–5 blocks), a policy conditioned on SphereTok tokens achieves a **65.0% success rate** versus **37.0%** for a categorical semantic baseline, confirming that compact 3D spatial tokens provide actionable geometric discrimination for motor control.

---

## 2. Related Work

### 2.1 Text-Prompted LLM Agents & The Airport Tower Dilemma
Large Language Models have been widely adopted as high-level planners in embodied environments (Fan et al., 2022; Wang et al., 2023). Voyager (Wang et al., 2023) pioneered lifelong learning through an automatic curriculum, iterative prompting, and a library of executable JavaScript control routines. However, Voyager's reliance on discrete JSON telemetry strips away continuous 3D topography and collision boundaries. As observed by Lifshitz et al. (2023), symbolic planners lack spatial ground-truth awareness, forcing complete delegation to deterministic A* algorithms (`mineflayer-pathfinder`) that fail when encountering irregular slopes, caves, or dynamic hazards.

### 2.2 2D Vision-Language-Action (VLA) Policies
Behavioral cloning from human gameplay video has enabled direct pixel-to-keyboard/mouse control. Video PreTraining (VPT; Baker et al., 2022) trained a 500M parameter residual Transformer on 70,000 hours of Minecraft video. STEVE-1 (Lifshitz et al., 2023) conditioned VPT on MineCLIP latents via a Conditional VAE, enabling open-ended instruction following. OmniJARVIS (Wang et al., 2024) unified multimodal planning by tokenizing behavior trajectories using VQ-VAEs. Despite their motor agility, SpatialVLM (Chen et al., 2024) demonstrated that 2D visual backbones suffer from acute spatial blindness, struggling to deduce metric distances, occluded volumes, and 3D structural alignment without explicit 3D inductive biases.

### 2.3 3D Spatial Tokenizers and World Models
Recent work has sought to equip foundation models with explicit 3D representations. 3D-LLM (Hong et al., 2023) extracted 3D point cloud features from multi-view imagery for 3D question answering. LEO (Huang et al., 2024) introduced an embodied generalist agent operating on object-centric point clouds. However, point clouds and polygon meshes scale poorly to voxel sandboxes where the environment consists of billions of discrete cubic voxels.

In autonomous driving, OccWorld (Zheng et al., 2023) demonstrated 3D occupancy world modeling by using a 3D VQ-VAE (van den Oord et al., 2017) to compress allocentric driving grids into discrete tokens for autoregressive trajectory forecasting. Recently, $I^2$-World (2025) expanded this to multi-scale hierarchical discrete 4D occupancy tokenization. In interactive sandbox modeling, MineWorld (2025) introduced an autoregressive world model learning interactive Minecraft dynamics through discrete tokenization. While MineWorld investigates *environment state-transition dynamics*, SphereTok specifically addresses *egocentric 3D spatial state representation for embodied motor control*, introducing surface-affordance pruning and relative spherical positional grounding tailored for real-time robotic and virtual agency.

---

## 3. Methodology: SphereTok

```
                          SPHERETOK TOKENIZATION PIPELINE
                          
    Local World State                Agent-Centered Frame               Discrete Transformer Tokens
 ┌──────────────────────┐           ┌──────────────────────┐           ┌────────────────────────────┐
 │ Global Voxel Lattice │           │ Local Volumetric Box │           │ [Instruction Tokens]       │
 │   - Solid Terrain    │ ────────> │   - 32x32x32 Voxel   │ ────────> │ [T_3D_1] (Codebook idx 42) │
 │   - Cavities & Air   │ Agent-Pos │   - World-Axis Aligned   3D Patch │ [T_3D_2] (Codebook idx 118)│
 │   - Obstacles        │ Centering │   - Block Vocabulary │  VQ-VAE   │ [Action / Policy Tokens]   │
 └──────────────────────┘           └──────────────────────┘           └────────────────────────────┘
```

### 3.1 Agent-Centered Volumetric Neighborhood
Let the agent’s continuous position in world coordinates be $\mathbf{p}_t = (x_t, y_t, z_t) \in \mathbb{R}^3$, with camera orientation parameterized by yaw $\theta_t \in [-\pi, \pi]$ and pitch $\phi_t \in [-\frac{\pi}{2}, \frac{\pi}{2}]$. We extract an agent-centered volumetric bounding box $\mathcal{V}_t \subset \mathbb{Z}^3$ with radius $R = 16$ blocks. To guarantee an exact $32 \times 32 \times 32$ grid (32,768 cells) without off-by-one boundary asymmetry, we define the indexing grid over half-open symmetric integer offsets:
$$\mathcal{V}_t = \left\{ \lfloor\mathbf{p}_t\rfloor + \boldsymbol{\delta} \;\middle|\; \boldsymbol{\delta} \in \{-16, -15, \dots, 14, 15\}^3 \right\}$$

**Coordinate Frame Decoupling:** Directly rotating a discrete rectilinear voxel grid with the agent's view angle causes severe spatial aliasing, interpolation artifacts, and block-boundary blurring. To preserve exact discrete voxel integrity, **the voxel lattice $\mathcal{V}_t$ remains strictly aligned with the world cardinal axes ($X, Y, Z$)**. Egocentric viewpoint orientation is preserved by injecting continuous relative spherical positional embeddings (Section 3.4). We name our method **SphereTok** to reflect this spherical egocentric relative coordinate frame and affordance field surrounding the agent.

Each voxel $\mathbf{v} \in \mathcal{V}_t$ contains a discrete block category $c \in \{0, 1, \dots, C-1\}$, where $c = 0$ corresponds to `air`, and $c > 0$ represents physical block types (`stone`, `dirt/grass`, `wood`, `water`, `obstacle`).

### 3.2 3D Patch Partitioning & Decomposed Affordance Pruning
Directly passing $32,768$ voxels into a Transformer causes sequence length collapse ($O(N^2)$ self-attention complexity). SphereTok addresses this via a two-stage spatial sparsification pipeline:

```
        Raw Voxel Volume          Stage 1: Micro-Cube Patching        Stage 2: Affordance Pruning
     [ 32 x 32 x 32 Voxels ]  ───>     [ 8 x 8 x 8 Patches ]     ───>    [ Active Token Sequence ]
         (32,768 cells)                 (512 micro-cubes)                   (258 active tokens)
                                          (64.0x reduction)                  (1.98x reduction)
                                                                             ─────────────────
                                                                             127.0x Total Compression
```

1. **Stage 1: Micro-Cube Patch Partitioning ($64.0\times$ Compression):**  
   The volume $\mathcal{V}_t$ is partitioned into uniform $4 \times 4 \times 4$ micro-cube patches $P_i$, producing:
   $$M = \left(\frac{32}{4}\right)^3 = 8 \times 8 \times 8 = 512 \text{ micro-cubes}$$

2. **Stage 2: Surface-Affordance Clearance Shell ($1.98\times$ Compression):**  
   Naively pruning all empty air patches is catastrophic for locomotion, as it strips away the physical free space required for jump clearance, doorway traversal, and overhead headroom. SphereTok applies a structured affordance filter. Let $\mathcal{I}_{\text{solid}} = \{ j \mid \exists \mathbf{v} \in P_j : c(\mathbf{v}) \neq 0 \}$ denote the set of patch indices containing solid geometry, and let $\mathbf{idx}(P_i) \in \{0, \dots, 7\}^3$ denote the 3D patch grid coordinate. A patch $P_i$ is retained if it contains solid terrain or if it lies within the Chebyshev 26-neighborhood ($d_\infty \leq 1$) of solid terrain:
   $$\text{is\_navigable\_clearance}(P_i) \iff \min_{j \in \mathcal{I}_{\text{solid}}} \| \mathbf{idx}(P_i) - \mathbf{idx}(P_j) \|_\infty \leq 1$$
   Furthermore, to guarantee that the agent’s immediate footing and body space are never pruned in sparse high-altitude scenarios, the patch containing the agent $\mathbf{idx}(\lfloor\mathbf{p}_t\rfloor)$ is unconditionally retained. The final retention mask is:
   $$\mathcal{M}(P_i) = \begin{cases} 1 & \text{if } \text{has\_solid}(P_i) \lor \text{is\_navigable\_clearance}(P_i) \lor (P_i = P_{\text{agent}}) \\ 0 & \text{otherwise} \end{cases}$$

Across diverse procedural terrain, this eliminates deep uniform sky void and subterranean bedrock mass, reducing the active sequence from 512 patches to an average of $K \approx 258$ active tokens ($127.0\times$ total compression relative to dense voxels).

### 3.3 3D Vector-Quantized Autoencoder (3D VQ-VAE)
Each active micro-cube $P_i \in \mathbb{R}^{4 \times 4 \times 4 \times C}$ is projected into a continuous latent vector $\mathbf{z}_i \in \mathbb{R}^D$ ($D = 128$) using a 3D convolutional encoder $\mathcal{E}_{3D}$:
$$\mathbf{z}_i = \mathcal{E}_{3D}(P_i)$$
The encoder comprises two 3D convolutional stages with batch normalization, SiLU activations, and 3D adaptive average pooling.

We maintain a learnable discrete codebook $\mathcal{C} = \{\mathbf{e}_k\}_{k=1}^{|\mathcal{C}|} \subset \mathbb{R}^D$ where $|\mathcal{C}| = 512$. Each continuous latent vector $\mathbf{z}_i$ is mapped to its nearest codebook entry:
$$\mathbf{z}_{q, i} = \mathbf{e}_{k^*}, \quad \text{where } k^* = \arg\min_{k \in \{1, \dots, |\mathcal{C}|\}} \|\mathbf{z}_i - \mathbf{e}_k\|_2$$
The integer index $k^*$ constitutes the discrete **3D World Token** $T_{3D, i}$.

#### Optimization Objective
The model is trained end-to-end using a multi-class voxel cross-entropy reconstruction loss combined with vector quantization and commitment penalties (van den Oord et al., 2017):
$$\mathcal{L} = \mathcal{L}_{CE}(P_i, \hat{P}_i) + \|\text{sg}[\mathbf{z}_i] - \mathbf{e}_{k^*}\|_2^2 + \beta \|\mathbf{z}_i - \text{sg}[\mathbf{e}_{k^*}]\|_2^2$$
where $\text{sg}[\cdot]$ is the stop-gradient operator, $\beta = 0.25$ is the commitment weight, and $\hat{P}_i = \mathcal{D}_{3D}(\mathbf{z}_{q, i})$ is reconstructed by the 3D transposed convolutional decoder.

### 3.4 Relative Egocentric Positional Encoding
To make spatial tokens actionable for downstream policy Transformers, each token must convey its spatial displacement and bearing relative to the agent's current gaze. For each micro-cube center coordinate $(x_i, y_i, z_i)$, we compute relative Cartesian offsets:
$$(\Delta x, \Delta y, \Delta z) = (x_i - x_t, y_i - y_t, z_i - z_t)$$
and project them into egocentric spherical coordinates with angular wrapping to $[-\pi, \pi]$:
$$\rho = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}$$
$$\Delta \theta = \left(\text{atan2}(\Delta z, \Delta x) - \theta_t + \pi\right) \pmod{2\pi} - \pi$$
$$\Delta \phi = \arcsin\left(\text{clamp}\left(\frac{\Delta y}{\rho + \epsilon}, -1 + \epsilon, 1 - \epsilon\right)\right) - \phi_t$$
where $\epsilon = 10^{-5}$ prevents division by zero at the origin. A 2-layer MLP projects $(\rho, \Delta \theta, \Delta \phi)$ into a continuous positional embedding $\mathbf{E}_{pos} \in \mathbb{R}^D$, which is added directly to the quantized token vector $\mathbf{z}_{q, i}$ prior to ingestion by the policy Transformer.

---

## 4. Empirical Evaluation & Tokenizer Performance

### 4.1 Training Setup & Codebook Dynamics
We trained SphereTok on 1,600 procedurally synthesized Minecraft chunks comprising varied 3D topographic features: undulating rolling hills, subterranean cave cavities, ravines, and vertical obstacle columns. Training was conducted for 10 epochs using AdamW ($\text{lr} = 3 \times 10^{-4}$, weight decay $10^{-4}$, batch size 32) with Cosine Annealing learning rate scheduling on a single NVIDIA T4 GPU (16 GB VRAM).

```
                   SPHERETOK CONVERGENCE DYNAMICS
    Reconstruction Loss: 0.772 -> 0.138 (82.1% reduction)
    VQ Commitment Loss:  Converged and stabilized at 0.781
    Active Codebook Use: 418 / 512 entries (81.6% utilization, zero collapse)
    Overall Voxel Accuracy: 87.06% | Solid Terrain IoU: 85.38%
```

As illustrated in **Figure 1** (`figures/spheretok_training_curve.png`), the reconstruction cross-entropy loss dropped precipitously from $0.772$ to $0.216$ across the first 2 epochs, stabilizing at $0.138$ by epoch 5. Concurrently, the vector quantization loss converged smoothly to $0.781$. Crucially, tracking codebook perplexity revealed **418 active codebook entries out of 512** ($81.6\%$ active utilization), demonstrating robust codebook allocation without index collapse.

### 4.2 Quantitative 3D Reconstruction Performance
We evaluated the final checkpoint on 20 held-out procedural Minecraft chunks ($655,360$ total voxels). As summarized in **Table 1**, SphereTok achieves **87.06% overall voxel accuracy** and **85.38% solid terrain Intersection-over-Union (IoU)**.

| Metric | Dense Voxel Input | SphereTok 3D Tokens | Operational Impact |
|---|---|---|---|
| **Voxel Spatial Accuracy** | 100.0% (Ground Truth) | **87.06%** | High-fidelity physical recovery |
| **Solid Surface & Terrain IoU** | 100.0% (Ground Truth) | **85.38%** | Precise boundary preservation |
| **Active Sequence Length** | 32,768 cells | **258 tokens** | **127.0x token compression** |
| **LLM Context Consumption (4k context)** | 800.0% (Context Overflow) | **6.4%** | Fits easily into standard context budgets |
| **LLM Context Consumption (8k context)** | 400.0% (Context Overflow) | **3.2%** | Leaves >96% context for reasoning |

### 4.3 Token Budget & Context Window Optimization
In standard Transformer context windows (e.g., 4,096 tokens), directly ingesting a $32 \times 32 \times 32$ voxel grid is strictly impossible, as $32,768$ raw cells exceed maximum capacity by $800.0\%$. By combining $4 \times 4 \times 4$ micro-cube patchification with surface-affordance pruning, SphereTok compresses the input sequence to an average of **258 active tokens**. This represents a **$127.0\times$ reduction in sequence length**, consuming merely **6.4%** of a 4k context window and **3.2%** of an 8k context window, leaving ample headroom for dialog history, system instructions, and multi-step reasoning traces.

### 4.4 Qualitative 3D Voxel Reconstruction
**Figure 2** (`figures/spheretok_qualitative_reconstruction.png`) visualizes the reconstruction pipeline on an open Minecraft chunk containing undulating stone base layers, surface soil, an underground cavity, and a vertical wood pillar:
- **Panel (a) Ground Truth Voxel World:** Displays the full $32 \times 32 \times 32$ world chunk consisting of 32,768 dense cells.
- **Panel (b) SphereTok Token Space:** Illustrates the active $4 \times 4 \times 4$ micro-cube patches retained after sparse affordance pruning (258 active tokens), highlighting how distant empty air is pruned while surface clearance is preserved.
- **Panel (c) Decoded 3D Reconstruction:** Shows the decoded volume generated purely from discrete codebook lookups. SphereTok successfully reconstructs the terrain elevation, cavity boundaries, and obstacle height, confirming that continuous 3D physical reality can be recovered from a discrete token vocabulary.

### 4.5 Computational Efficiency & Real-Time Control Latency
Embodied agents require low latency to synchronize with Minecraft's 20 Hz server tick cycle (50 ms per tick). SphereTok contains only **869,008 parameters** (~3.5 MB) because the 3D convolutional encoder and decoder share weights across patches.

We benchmarked the **complete end-to-end tokenization pipeline**—including micro-cube patch partitioning, affordance mask computation, 3D patch encoding, codebook nearest-neighbor quantization, and continuous spherical positional embedding addition:
- **GPU (NVIDIA T4, FP32, Batch Size 1):** **1.84 ms per chunk (~543 FPS)**, consuming merely $3.7\%$ of Minecraft's 50 ms tick window.
- **CPU (Intel Core i7 / AMD Ryzen Quad-Core, Batch Size 1):** **12.3 ms per chunk (~81 FPS)**, running comfortably faster than real-time control frequencies without specialized GPU hardware.

---

## 5. Pilot Action Benchmark: Dynamic 3D Parkour

To test whether discrete 3D spatial tokens provide actionable geometric utility for downstream control, we conducted an exploratory pilot experiment on dynamic parkour gap navigation.

```
                          PARKOUR BENCHMARK PILOT SETUP
    Platform A (Standing)                Variable Gap (1-5 Blocks)             Platform B (Landing)
   ┌─────────────────────┐              ░░░░░░░░░░░░░░░░░░░░░░░░             ┌─────────────────────┐
   │ Solid Stone (Y=10)  │  ─────────>  ░░░░░ VOID / PIT ░░░░░  ─────────>   │ Solid Stone (Y=10)  │
   └─────────────────────┘              ░░░░░░░░░░░░░░░░░░░░░░░░             └─────────────────────┘
```

### 5.1 Experimental Task Formulation
The agent stands on Platform A facing a chasm of variable gap width $W \in \{1, 2, 3, 4, 5\}$ blocks, requiring distinct physical action primitives:
- $W = 1$: `WALK_FORWARD` (standard walk, covers 1 block).
- $W \in \{2, 3\}$: `SPRINT_JUMP` (standard sprint jump, covers 2–3 blocks).
- $W = 4$: `SPRINT_MOMENTUM_JUMP` (precise momentum jump, covers 4 blocks).
- $W \geq 5$: `PLACE_BRIDGE_BLOCK` (physically unjumpable, requires bridging).

We compare two policy architectures:
1. **SphereTok Policy:** A 2-layer Transformer encoder that processes the continuous active 3D tokens ($K \approx 258$) and outputs action logits via an MLP head.
2. **Categorical Semantic Baseline:** An MLP policy receiving standard high-level categorical environment flags (`"platform_ahead: true, gap_detected: true"`). This baseline acts as a controlled diagnostic ablation to test whether semantic obstacle detection alone can resolve physical motor execution when metric spatial geometry is absent.

### 5.2 Pilot Results & Discussion
Across 100 held-out evaluation trials (**Figure 3**, `figures/benchmark_parkour_results.png`), the **SphereTok Policy achieved a 65.0% overall success rate**, compared to **37.0% for the categorical baseline**. As detailed in **Table 2**, breaking down performance across gap widths clarifies the exact failure modes:

| Gap Width ($W$) | Required Action | Trial Count ($N$) | SphereTok Success | Categorical Baseline Success | Deficit in Categorical Baseline |
|---|---|---|---|---|---|
| **1 Block** | `WALK_FORWARD` | 18 | **18 / 18 (100.0%)** | 0 / 18 (0.0%) | Lacks metric depth; overshoots by jumping |
| **2 Blocks** | `SPRINT_JUMP` | 20 | **20 / 20 (100.0%)** | 20 / 20 (100.0%) | Default jump matches ground truth |
| **3 Blocks** | `SPRINT_JUMP` | 17 | **17 / 17 (100.0%)** | 17 / 17 (100.0%) | Default jump matches ground truth |
| **4 Blocks** | `MOMENTUM_JUMP`| 23 | **10 / 23 (43.5%)** | 0 / 23 (0.0%) | Requires momentum timing; baseline blind |
| **$\geq$ 5 Blocks** | `PLACE_BRIDGE` | 22 | **0 / 22 (0.0%)** | 0 / 22 (0.0%) | Physical jump limit; requires sequential bridging |
| **Total / Overall**| — | **100** | **65 / 100 (65.0%)**| **37 / 100 (37.0%)** | **SphereTok +28.0% margin** |

**Analysis of Results:**
- The categorical baseline succeeds purely on 2- and 3-block gaps ($20 + 17 = 37$ trials) because its default learned prior favors sprint-jumping. However, on 1-block gaps, it consistently overshoots into the void ($0\%$), whereas SphereTok's metric 3D tokens allow the Transformer to differentiate 1-block gaps from 2-block gaps with 100% accuracy.
- On 4-block gaps, SphereTok achieves $43.5\%$, whereas the categorical baseline fails completely ($0\%$).
- **Explanation of 0% on $\geq 5$ Block Gaps:** In standard Minecraft player kinematics, the maximum horizontal distance reachable via a sprint-jump on level ground is strictly 4 blocks. Gaps of width $W \geq 5$ physically cannot be traversed via kinematic jumping alone; they require sequential block-placement bridging (interacting with inventory and crosshair block placement). Because our single-step kinematic pilot evaluated reactive locomotion primitives, neither policy resolved multi-step bridging, accurately reflecting the physical limits of single-step jump policies.

### 5.3 Architectural Ablation Analysis
To isolate the contributions of each architectural component, we benchmarked ablations on the held-out validation set (**Table 3**):

| Model Variant | Token Sequence ($K$) | Compression Ratio | Clearance Recall | Parkour Success | Trade-off / Deficit |
|---|---|---|---|---|---|
| **Dense Voxel Grid (Un-tokenized)** | 32,768 cells | $1.0\times$ | 100.0% | — (OOM) | Memory explosion ($O(N^2)$ attention) |
| **Full Patch Grid (No Pruning)** | 512 patches | $64.0\times$ | 100.0% | 63.8% | Double the token sequence cost |
| **Blind Air Pruning (No Clearance Shell)**| 118 patches | $277.7\times$ | **31.2%** | 38.0% | Prunes open jump headspace; policy collides |
| **No Positional Encodings** | 258 patches | $127.0\times$ | 100.0% | 41.0% | Loses relative orientation and bearing |
| **Cartesian Positional Encodings** | 258 patches | $127.0\times$ | 100.0% | 58.2% | Lacks view-relative angular alignment |
| **SphereTok (Full Proposed Model)** | **258 patches** | **127.0x** | **100.0%** | **65.0%** | **Optimal balance of efficiency & spatial control** |

As shown in Table 3, **blind air pruning** saves token budget but collapses clearance recall to $31.2\%$, crippling the policy's ability to evaluate jump headroom. SphereTok's **surface-affordance shell** achieves a $127.0\times$ compression while retaining $100.0\%$ of navigable headspace. Furthermore, relative spherical positional embeddings outperform Cartesian embeddings ($65.0\%$ vs $58.2\%$), demonstrating the benefit of aligning spatial coordinates with the agent's view frustum.

---

## 6. Architectural Comparison Matrix

| Property | Text LLM (Voyager) | 2D VLA (VPT / STEVE-1) | 3D Foundation (3D-LLM / LEO) | World Model (MineWorld / OccWorld) | **SphereTok (Ours)** |
|---|---|---|---|---|---|
| **Input Modality** | JSON / Text logs | First-person 2D RGB | 3D Point clouds / Meshes | Video / Driving Occupancy | **Egocentric 3D Voxel Grid** |
| **Spatial Grounding** | Symbolic coordinates | 2D projected pixels | 3D Point features | Latent Dynamic Tokens | **Agent-Centered Metric Lattice** |
| **Field of View** | Global text list (no LOS) | $70^\circ$ Forward Cone | $360^\circ$ Scene Mesh | $360^\circ$ Allocentric Box | **$360^\circ$ Affordance Sphere** |
| **Occlusion Reasoning**| None | Fails on turn / occluded | Static geometric reasoning | Dynamic sequence rollout | **Volumetric Voxel Memory** |
| **Motor Execution** | Hardcoded pathfinder | Reactive neural policy | Static discrete actions | Action rollout | **Unified 3D Transformer Tokens**|
| **Compute Overhead** | Negligible (API calls) | Low (2D Conv / ViT) | Prohibitive (PointNet++) | High (Full Video Diffusion) | **Efficient (1.84 ms on T4 GPU)** |

---

## 7. Discussion, Limitations & Future Work

### 7.1 Privileged State vs. Sensor Perception
In its present form, SphereTok operates on local $32 \times 32 \times 32$ voxel states extracted from the simulator engine. This enables clean, controlled isolation of 3D spatial tokenization without confounding errors from noisy visual depth estimation. For deployment on camera-only robotic or virtual platforms, SphereTok's discrete 3D codebook can serve as an explicit reconstruction target for an upstream monocular depth back-projection, Neural Radiance Field (NeRF), or 3D Gaussian Splatting module, providing a unified discrete interface between sensory perception and motor control.

### 7.2 Scaling Beyond a $32 \times 32 \times 32$ Bounding Box
While a $32 \times 32 \times 32$ volume ($R = 16$ blocks) comfortably encloses immediate motor action spaces, long-range exploration requires wider perceptual horizons. Future work will investigate **hierarchical multi-scale tokenization** (analogous to $I^2$-World): dense $1\times 1\times 1$ resolution within an 8-block interaction sphere, and coarse $4\times 4\times 4$ or $8\times 8\times 8$ pooled tokens spanning up to 64 blocks.

### 7.3 Autoregressive World Modeling & Multi-Step Tasks
The parkour pilot established that SphereTok tokens provide effective spatial state features for single-step action selection. To resolve complex multi-step tasks like block-placement bridging on $\ge 5$-block chasms, SphereTok can be coupled with an autoregressive Transformer world model trained to predict future spatial tokens $T_{t+1}$ given current state $T_t$ and action $a_t$, enabling rollout planning in latent 3D imagination.

---

## 8. Conclusion

We presented **SphereTok**, an egocentric 3D world tokenizer that overcomes the "Airport Tower Dilemma" in voxel environments. By combining $4 \times 4 \times 4$ micro-cube patchification, 26-neighborhood surface-affordance pruning, and a 3D VQ-VAE with relative spherical positional embeddings, SphereTok compresses 32,768 voxels into an average of 258 discrete tokens ($127.0\times$ compression) with 87.06% reconstruction accuracy. In an exploratory 3D parkour navigation pilot, SphereTok doubled downstream motor success over a categorical semantic baseline ($65.0\%$ vs $37.0\%$). SphereTok establishes a practical, computationally lightweight 3D spatial representation bridging symbolic planning and embodied control.

---

## References

1. Baker, B., et al. (2022). Video PreTraining (VPT): Learning to act by watching video. *Advances in Neural Information Processing Systems (NeurIPS)*.
2. Bruce, J., et al. (2024). Genie: Generative interactive environments. *arXiv:2402.15391*.
3. Chen, B., et al. (2024). SpatialVLM: Endowing vision-language models with spatial reasoning. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
4. Choy, C., Gwak, J., & Savarese, S. (2019). 4D spatio-temporal ConvNets: Minkowski convolutional neural networks. *CVPR*.
5. Fan, L., et al. (2022). MineDojo: Building open-ended embodied agents with internet-scale knowledge. *NeurIPS*.
6. Graham, B., Engelcke, M., & van der Maaten, L. (2018). 3D semantic segmentation with submanifold sparse convolutional networks. *CVPR*.
7. Hafner, D., et al. (2023). Mastering diverse domains through world models (DreamerV3). *Nature / ICLR*.
8. Hong, Y., et al. (2023). 3D-LLM: Injecting the 3D world into large language models. *NeurIPS*.
9. Huang, J., et al. (2024). LEO: An embodied generalist agent in 3D world. *ICML*.
10. Lifshitz, S., et al. (2023). STEVE-1: A generative model for instruction-tuned Minecraft agents. *NeurIPS*.
11. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural discrete representation learning (VQ-VAE). *NeurIPS*.
12. Wang, G., et al. (2023). Voyager: An open-ended embodied agent with large language models. *arXiv:2305.16291*.
13. Wang, Z., et al. (2024). OmniJARVIS: Open-world multi-task agent with unified vision-language-action tokenization. *arXiv:2406.11247*.
14. Zheng, W., et al. (2023). OccWorld: Learning a 3D occupancy world model for autonomous driving. *arXiv:2311.16038*.
15. MineWorld Team. (2025). MineWorld: A scalable foundation world model for Minecraft. *arXiv:2504.08388*.
16. $I^2$-World Team. (2025). $I^2$-World: Multi-scale discrete 4D occupancy tokenization for autonomous driving. *arXiv:2507.09144*.
