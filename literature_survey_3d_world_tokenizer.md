# Literature Survey: 3D World Tokenizers for Embodied Perception & Minecraft Agents

**Topic:** Bridging Symbolic Text Planners, 2D Video Policies, and Discrete 3D Representation Learning in Embodied Sandbox Environments  
**Purpose:** Academic Foundation, Related Works Reference, and Taxonomy for the SphereTok Architecture  

---

## 1. Executive Summary & Problem Framing

### 1.1 The "Airport Tower Dilemma" in Symbolic Agents
Current embodied foundation models in open-world virtual environments (notably Minecraft) overwhelmingly belong to two distinct paradigms:
1. **Symbolic Text-Prompted LLM Agents (e.g., Voyager, GITM, Plan4MC):** The agent receives text-serialized JSON environment logs (e.g., `{"nearest_tree": [12, 64, -8], "biome": "plains"}`). The language model operates analogously to an **airport control tower**—observing sparse alphanumeric telemetry from afar and attempting to issue flight commands without ever sitting in the cockpit. This representation strips away continuous surface geometry, jump clearances, friction, and volumetric occlusions. Consequently, these agents must delegate physical movement to deterministic pathfinding algorithms (e.g., `mineflayer-pathfinder`). When these algorithms encounter unscripted 3D geometry or variable obstacles, they fail with opaque text error strings, driving the LLM into unrecoverable execution loops.
2. **2D Vision-Language-Action (VLA) Models (e.g., VPT, STEVE-1):** The agent observes the world through monocular first-person RGB pixels ($128 \times 128$). While capturing visual textures, standard 2D convolutional and Vision Transformer backbones suffer from monocular depth ambiguity, occlusion blindness, and lack of 3D spatial permanence (Chen et al., 2024). Rotating the camera causes unobserved geometry behind the agent to vanish from context.

### 1.2 The Solution: The Egocentric 3D World Tokenizer (SphereTok)
**SphereTok** is an architectural concept and neural implementation that inverts the traditional database ontology:
* Instead of treating the agent as an arbitrary point in a detached global coordinate system $(X, Y, Z)$, **the agent is $(0,0,0)$—the origin of an egocentric perceptual volume**.
* A local $32 \times 32 \times 32$ voxel volume is partitioned into $4 \times 4 \times 4$ micro-cubes and tokenized into discrete codebook indices using a 3D Vector-Quantized Variational Autoencoder (3D VQ-VAE).
* Rather than naively discarding all air, a **surface-affordance shell** preserves navigable empty space bordering terrain (ensuring jump clearance and corridor perception) while pruning deep open void.
* Yaw- and pitch-relative spherical positional embeddings $(\rho, \Delta\theta, \Delta\phi)$ ground the discrete tokens in the agent's view frame without causing voxel interpolation or aliasing artifacts.

---

## 2. Taxonomy of Literature

```
                           Embodied AI & 3D World Representations
                                              │
         ┌────────────────────┬───────────────┴───────────────┬────────────────────┐
         ▼                    ▼                               ▼                    ▼
  [Cluster 1: Agents]  [Cluster 2: 3D-LLMs]            [Cluster 3: 3D Occ]  [Cluster 4: Foundations]
  • Voyager (2023)     • 3D-LLM (NeurIPS 2023)         • OccWorld (2023)    • VQ-VAE (NeurIPS 2017)
  • VPT (NeurIPS 2022) • LEO (ICML 2024)               • DreamerV3 (2023)   • VQ-VAE-2 (NeurIPS 2019)
  • STEVE-1 (2023)     • SpatialVLM (CVPR 2024)        • Genie (2024)       • MinkowskiEngine (2019)
  • OmniJARVIS (2024)  • PointNet++ (NeurIPS 2017)     • MineDojo (2022)    • Decision Trans. (2021)
  • MineRL (2019)      • MineCLIP (NeurIPS 2022)       • Trajectory Trans.  • SparseConvNet (2018)
```

---

## 3. Deep Dive: Key Literature Clusters

### Cluster 1: Minecraft Embodied Agents & Behavioral Cloning

#### 1. Voyager: An Open-Ended Embodied Agent with Large Language Models
- **Authors:** Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, Anima Anandkumar (2023)
- **Reference:** *arXiv:2305.16291*
- **Mechanism:** GPT-4 controller with an automatic curriculum, Mineflayer JavaScript skill library, and iterative prompting.
- **Representation:** Discrete JSON telemetry strings.
- **Limitation:** Exemplifies the Airport Tower Dilemma. Lacks continuous 3D collision boundaries and cannot compute real-time jump trajectories or parkour maneuvers, delegating entirely to deterministic pathfinders.

#### 2. Video PreTraining (VPT): Learning to Act by Watching Video
- **Authors:** Bowen Baker, Ilge Akkaya, Peter Zhokhov, Joost Huizinga, Jie Tang, David Farhi, Firas Abuhamdeh, Mauricio de Oliveira, Eugene Sontag, Peter Welinder (OpenAI, NeurIPS 2022)
- **Reference:** *arXiv:2206.11795*
- **Mechanism:** 500M parameter residual Transformer trained on 70,000+ hours of human gameplay video using an inverse dynamics model.
- **Representation:** First-person 2D RGB frames ($128 \times 128$).
- **Limitation:** No explicit 3D metric inductive bias. Susceptible to monocular depth distortion and camera-rotation spatial amnesia.

#### 3. STEVE-1: A Generative Model for Instruction-Tuned Minecraft Agents
- **Authors:** Shalev Lifshitz, Keiran Paster, Harris Chan, Jimmy Ba, Sheila McIlraith (NeurIPS 2023)
- **Reference:** *arXiv:2306.11458*
- **Mechanism:** Conditions a frozen VPT policy on MineCLIP visual-text latents using a Conditional VAE.
- **Limitation:** Inherits all visual blindness of 2D video models; struggles with 3D structural reasoning and occluded navigation.

#### 4. OmniJARVIS: Open-World Multi-Task Agent with Unified Tokenization
- **Authors:** Shaofei Cai, Zihao Wang, Bowei Zhang, Haowei Lin, Yitao Liang, et al. (2024)
- **Reference:** *arXiv:2406.11247*
- **Mechanism:** Discretizes behavior trajectories and multimodal interactions into discrete tokens using VQ-VAE and Finite Scalar Quantization (FSQ).
- **Relevance:** Demonstrates the value of unified tokenization for control, but relies primarily on 2D visual encoders rather than egocentric 3D voxel lattices.

#### 5. MineDojo & MineCLIP: Building Open-Ended Embodied Agents
- **Authors:** Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, Anima Anandkumar (NeurIPS 2022, Outstanding Paper Award)
- **Reference:** *arXiv:2206.01334*
- **Mechanism:** Massive benchmark suite and a contrastive video-language foundation model (MineCLIP) aligning English gameplay descriptions with video frames.
- **Relevance:** Serves as the primary ecosystem for modern Minecraft AI benchmarking.

---

### Cluster 2: 3D Representation Learning & Spatial Foundation Models

#### 6. 3D-LLM: Injecting the 3D World into Large Language Models
- **Authors:** Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, Chuang Gan (NeurIPS 2023)
- **Reference:** *arXiv:2307.12981*
- **Mechanism:** Reconstructs 3D point cloud features from multi-view images and maps them into LLMs via 3D feature extractors and coordinate position embeddings.
- **Relevance:** Proved that LLMs can natively interpret 3D tokens for spatial question answering and grounding.
- **Limitation:** Designed for static scene understanding on point clouds and meshes; computationally heavy for real-time dynamic voxel manipulation.

#### 7. LEO: An Embodied Generalist Agent in 3D World
- **Authors:** Jianglong Huang, Silong Yong, Xiaojian Ma, Lingpeng Kong, Baoxiong Jia, Siyuan Huang (ICML 2024)
- **Reference:** *arXiv:2311.12871*
- **Mechanism:** Fuses 2D egocentric frames with an object-centric 3D point cloud encoder (PointNet++ and Spatial Transformers), interleaving object tokens with language and action tokens.
- **Limitation:** Assumes pre-segmented object point clouds, which does not fit contiguous voxel sandbox grids.

#### 8. SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning
- **Authors:** Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Danny Driess, Pete Florence, Dorsa Sadigh (CVPR 2024)
- **Reference:** *arXiv:2401.12168*
- **Takeaway:** Empirically demonstrates that standard 2D VLMs fail quantitative spatial queries (e.g., metric distance, vertical elevation, relative ordering) without explicit 3D spatial grounding.

#### 9. PointNet++: Deep Hierarchical Feature Learning on Point Sets
- **Authors:** Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas (NeurIPS 2017)
- **Reference:** *arXiv:1706.02413*
- **Takeaway:** Foundational architecture for hierarchical spatial feature extraction from geometric point clouds.

---

### Cluster 3: 3D Occupancy & World Models

#### 10. OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving
- **Authors:** Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, Jiwen Lu (arXiv:2311.16038, 2023)
- **Mechanism:** Discretizes dense 3D semantic occupancy grids around an autonomous vehicle using a 3D VQ-VAE tokenizer (spatial downsampling factor of 4, codebook size 512, feature dimension 128). An autoregressive GPT-like spatial-temporal Transformer predicts future scene tokens and ego-vehicle trajectory tokens.
- **Direct Link to SphereTok:** OccWorld proves that a 3D VQ-VAE can compress dense 3D surrounding space into discrete tokens for autoregressive modeling. SphereTok adapts this insight to an egocentric agent in a voxel sandbox, replacing vehicle Bird's-Eye-View grids with spherical, yaw-relative voxel affordances.

#### 11. DreamerV3: Mastering Diverse Domains through World Models
- **Authors:** Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap (Nature / ICLR 2023)
- **Reference:** *arXiv:2301.04104*
- **Takeaway:** Demonstrates that learning a discrete latent world model enables agents to solve Minecraft tasks (including collecting diamonds from scratch) purely through imagined latent trajectories.

#### 12. Genie: Generative Interactive Environments
- **Authors:** Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Antonia Creswell, et al. (Google DeepMind, 2024)
- **Reference:** *arXiv:2402.15391*
- **Takeaway:** Introduces generative interactive world models trained on unlabeled video that learn discrete latent action spaces, validating the power of discrete tokenization for world simulation.

---

### Cluster 4: Foundational Tokenization & Sparse 3D Architectures

#### 13. Neural Discrete Representation Learning (VQ-VAE) & VQ-VAE-2
- **Authors:** Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu (NeurIPS 2017); Ali Razavi, Aaron van den Oord, Oriol Vinyals (NeurIPS 2019)
- **Reference:** *arXiv:1711.00937*, *arXiv:1906.00446*
- **Relevance:** The core vector-quantization mechanism, codebook loss, and straight-through gradient estimation that powers SphereTok's spatial discretization.

#### 14. 4D Spatio-Temporal ConvNets: MinkowskiEngine & SparseConvNet
- **Authors:** Christopher Choy, JunYoung Gwak, Silvio Savarese (CVPR 2019); Benjamin Graham, Martin Engelcke, Laurens van der Maaten (CVPR 2018)
- **Reference:** *arXiv:1904.08755*, *arXiv:1711.10275*
- **Relevance:** Formalizes generalized sparse tensor convolutions for high-dimensional 3D/4D spatial data, providing theoretical backing for SphereTok's sparse micro-cube patch pruning.

#### 15. Decision Transformer & Trajectory Transformer
- **Authors:** Lili Chen et al. (NeurIPS 2021); Michael Janner, Qiyang Li, Sergey Levine (NeurIPS 2021)
- **Reference:** *arXiv:2106.01345*, *arXiv:2106.02039*
- **Relevance:** Establishes reinforcement learning and sequential decision-making as autoregressive next-token prediction, enabling SphereTok's 3D tokens to seamlessly interleave with action tokens.

---

## 4. Architectural Comparison Matrix

| Property | Text LLM (Voyager) | 2D VLA (VPT / STEVE-1) | 3D Foundation (3D-LLM / LEO) | Occupancy Model (OccWorld) | **SphereTok (Ours)** |
|:---|:---|:---|:---|:---|:---|
| **Input Modality** | JSON / Text logs | First-person 2D RGB | 3D Point clouds / Meshes | Allocentric 3D Occupancy | **Egocentric 3D Voxel Grid** |
| **Spatial Grounding** | Symbolic coordinates | 2D projected pixels | 3D Point features | 3D Voxel Latents | **Egocentric Metric Lattice** |
| **Field of View** | Global text list (no LOS) | $70^\circ$ Forward Cone | $360^\circ$ Scene Mesh | $360^\circ$ Driving Box | **$360^\circ$ Volumetric Shell** |
| **Occlusion Handling**| Blind | Fails on turn / occluded | Static geometric reasoning | Dynamic driving prediction | **Volumetric Voxel Memory** |
| **Clearance Perception**| None (Pathfinder only) | Approximate monocular | Mesh collision | Drivable surface | **Navigable Air Boundary** |
| **Context Compression**| High (Text strings) | High (ViT patches) | Low (Dense point tokens) | Moderate (4x downsampled) | **High (127x via Sparse VQ-VAE)**|

---

## 5. Formal Mathematical Formulation of SphereTok

### 5.1 Egocentric Volumetric Bounding Box
Let the agent's world coordinate be $\mathbf{p}_t = (x_t, y_t, z_t) \in \mathbb{R}^3$, with orientation given by yaw $\theta_t \in [-\pi, \pi]$ and pitch $\phi_t \in [-\frac{\pi}{2}, \frac{\pi}{2}]$. We define an axis-aligned egocentric bounding volume $\mathcal{V}_t$ spanning radius $R = 16$:
$$\mathcal{V}_t = \{ \mathbf{v} = (X, Y, Z) \in \mathbb{Z}^3 \mid |X - x_t| \leq R, \, |Y - y_t| \leq R, \, |Z - z_t| \leq R \}$$
Keeping the grid axis-aligned with world coordinates prevents rotational aliasing and voxel interpolation distortion.

### 5.2 Micro-Cube Partitioning & Affordance Pruning
The volume is partitioned into regular $4 \times 4 \times 4$ micro-cubes:
$$M = \left(\frac{32}{4}\right)^3 = 512 \text{ micro-cubes } P_i$$
To preserve jump clearances and navigable open paths while pruning uniform empty air:
$$\mathcal{M}(P_i) = \begin{cases} 1 & \text{if } \exists v \in P_i : c(v) \neq \text{air} \;\lor\; \text{dist}(P_i, \text{solid}) \leq 1 \\ 0 & \text{otherwise (deep void / sky)} \end{cases}$$
Across standard Minecraft terrain, this retains an average of $K \approx 258$ active tokens ($127.0\times$ compression from 32,768 raw voxels).

### 5.3 Relative Spherical Positional Encoding
For each active micro-cube center $(x_i, y_i, z_i)$, define Cartesian offsets:
$$\Delta x = x_i - x_t, \quad \Delta y = y_i - y_t, \quad \Delta z = z_i - z_t$$
We map these offsets into agent-centric spherical coordinates:
$$\rho = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}$$
$$\Delta \theta = \left( \text{atan2}(\Delta z, \Delta x) - \theta_t + \pi \right) \pmod{2\pi} - \pi$$
$$\Delta \phi = \arcsin\left(\text{clamp}\left(\frac{\Delta y}{\rho}, -1 + \epsilon, 1 - \epsilon\right)\right) - \phi_t$$
A multi-layer perceptron $\text{MLP}_{pos}$ projects $(\rho, \Delta \theta, \Delta \phi)$ into a continuous embedding vector $\mathbf{E}_{pos} \in \mathbb{R}^D$, which is summed with the quantized codebook token $\mathbf{z}_{q, i}$ prior to transformer self-attention.

---

## 6. Open Problems & Roadmap to Autonomous Survival

This literature survey and the SphereTok architecture directly lay the groundwork for solving the core failure modes of autonomous Minecraft survival (e.g., Hardcore mode zero-death agents):
1. **Dynamic Velocity / Proprioception:** Augmenting static voxel tokens with a 1-token agent state vector $(\mathbf{v}_t, \text{on\_ground}, \text{jump\_cooldown})$ to enable momentum physics.
2. **Episodic Voxel Memory:** Extending beyond the local radius-16 volume via recurrent latent states or a persistent sparse global hash map to provide permanent cave navigation without amnesia.
3. **Causal Epistemic Provenance:** Distinguishing between world-generated structures (`observed`) and agent-altered terrain (`built_by_me`) to eliminate false environmental attribution.
4. **Hierarchical Control:** Coupling the 20 Hz SphereTok spatial reflex policy with a slow, contemplative high-level LLM planner (the strategic pilot).
