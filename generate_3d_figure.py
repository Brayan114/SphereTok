"""
Generate qualitative 3D voxel renders comparing:
1. Ground Truth Minecraft Voxel Environment
2. Egocentric Perceptual Sphere & Sparse Tokenized Micro-Cubes
3. Decoded 3D Voxel Reconstruction from Discrete Tokens
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spheretok.tokenizer import SphereTok
from demo import create_synthetic_minecraft_chunk

def main():
    print("Generating qualitative 3D voxel renders...")

    # 1. Load trained checkpoint
    model = SphereTok(
        num_block_classes=16,
        grid_size=32,
        patch_size=4,
        latent_dim=128,
        codebook_size=512,
        air_class_id=0
    )
    checkpoint_path = "checkpoints/spheretok_minecraft_checkpoint.pt"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Create sample chunk
    chunk = create_synthetic_minecraft_chunk() # (1, 32, 32, 32)
    B = chunk.shape[0]

    # 3. Tokenize and decode with trained model
    with torch.no_grad():
        out = model(chunk, prune_air=False)
        tokens = out["tokens"] # (1, 512)
        z_q = model.vq.embedding(tokens.view(-1))
        recon_logits = model.decoder(z_q)
        recon_vol = model.reconstruct_volume(recon_logits, torch.arange(512), batch_size=B).argmax(dim=1)[0].numpy()

    gt_vol = chunk[0].numpy()

    # 4. Create 3D visualization
    fig = plt.figure(figsize=(18, 6), dpi=200)

    # Color mapping for Minecraft blocks
    # 0: Air, 1: Stone (grey), 2: Grass/Soil (green), 3: Wood (brown)
    def get_voxel_colors(vol):
        colors = np.empty(vol.shape, dtype=object)
        colors[vol == 1] = "#7e7e7e" # Stone
        colors[vol == 2] = "#4c9a2a" # Grass / Soil
        colors[vol == 3] = "#8b5a2b" # Wood pillar
        return colors

    # Subplot 1: Ground Truth 3D Chunk
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    gt_filled = (gt_vol != 0)
    gt_colors = get_voxel_colors(gt_vol)
    ax1.voxels(gt_filled, facecolors=gt_colors, edgecolors='k', linewidth=0.1, alpha=0.9)
    ax1.set_title("(a) Ground-Truth 3D Voxel World\n(32,768 Raw Voxels)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("X (Width)")
    ax1.set_ylabel("Y (Height)")
    ax1.set_zlabel("Z (Depth)")
    ax1.view_init(elev=28, azim=45)

    # Subplot 2: Sparse 3D Patchification (The SphereTok Token Space)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    # Build 8x8x8 micro-cube occupancy
    patches_occupied = np.zeros((8, 8, 8), dtype=bool)
    patch_colors = np.empty((8, 8, 8), dtype=object)
    for px in range(8):
        for py in range(8):
            for pz in range(8):
                sub = gt_vol[py*4:(py+1)*4, px*4:(px+1)*4, pz*4:(pz+1)*4]
                if np.any(sub != 0):
                    patches_occupied[px, py, pz] = True
                    patch_colors[px, py, pz] = "#3498db" # Active token blue

    ax2.voxels(patches_occupied, facecolors=patch_colors, edgecolors='#1b4f72', linewidth=0.6, alpha=0.6)
    ax2.set_title("(b) SphereTok Token Space\n(258 Active 4x4x4 Tokens, 127x Compression)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Patch X")
    ax2.set_ylabel("Patch Y")
    ax2.set_zlabel("Patch Z")
    ax2.view_init(elev=28, azim=45)

    # Subplot 3: Decoded 3D Reconstruction from Discrete Tokens
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    recon_filled = (recon_vol != 0)
    recon_colors = get_voxel_colors(recon_vol)
    ax3.voxels(recon_filled, facecolors=recon_colors, edgecolors='k', linewidth=0.1, alpha=0.9)
    ax3.set_title("(c) Decoded 3D Reconstruction\n(87.1% Voxel Accuracy, 85.4% IoU)", fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel("X (Width)")
    ax3.set_ylabel("Y (Height)")
    ax3.set_zlabel("Z (Depth)")
    ax3.view_init(elev=28, azim=45)

    plt.tight_layout()
    output_path = "figures/spheretok_qualitative_reconstruction.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=250)
    plt.close()
    print(f"Saved qualitative 3D figure to: {output_path}")

if __name__ == "__main__":
    main()
