"""
SphereTok Demo: Testing the 3D World Tokenizer on a simulated Minecraft terrain chunk.
"""

import torch
from spheretok.tokenizer import SphereTok

def create_synthetic_minecraft_chunk():
    """
    Creates a 32x32x32 synthetic Minecraft chunk:
    - 0: Air
    - 1: Stone
    - 2: Dirt / Grass
    - 3: Oak Wood
    - 4: Water
    """
    grid = torch.zeros((1, 32, 32, 32), dtype=torch.long)

    # 1. Ground bedrock / stone layers (Y from 0 to 10)
    grid[:, 0:10, :, :] = 1

    # 2. Surface dirt / grass layers (Y from 10 to 14)
    grid[:, 10:14, :, :] = 2

    # 3. Carve an underground cave cavity (radius 4 at X=16, Y=7, Z=16)
    for y in range(4, 10):
        for x in range(12, 20):
            for z in range(12, 20):
                if (x - 16)**2 + (y - 7)**2 + (z - 16)**2 <= 16:
                    grid[:, y, x, z] = 0 # Air inside cave

    # 4. Add a parkour obstacle / tree trunk on the surface (Y from 14 to 22 at X=16, Z=16)
    grid[:, 14:22, 16, 16] = 3

    return grid

def main():
    print("=" * 70)
    print("  SphereTok: Egocentric 3D World Tokenizer Demonstration")
    print("=" * 70)

    # Initialize SphereTok
    model = SphereTok(
        num_block_classes=16,
        grid_size=32,
        patch_size=4,
        latent_dim=128,
        codebook_size=512,
        air_class_id=0
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Init] Trainable Parameters: {total_params:,}")

    # Generate Minecraft chunk
    voxel_chunk = create_synthetic_minecraft_chunk()
    total_voxels = voxel_chunk.numel()
    solid_voxels = (voxel_chunk != 0).sum().item()
    air_voxels = (voxel_chunk == 0).sum().item()

    print(f"[Input Chunk] Total Voxel Volume: {total_voxels:,} blocks (32 x 32 x 32)")
    print(f"              Solid Terrain / Objects: {solid_voxels:,} ({solid_voxels / total_voxels * 100:.1f}%)")
    print(f"              Empty Air: {air_voxels:,} ({air_voxels / total_voxels * 100:.1f}%)")

    # Run SphereTok forward pass
    print("\n[Tokenizing] Passing through Egocentric 3D Patch VQ-VAE...")
    output = model(voxel_chunk, yaw=0.785, pitch=0.2, prune_air=True)

    tokens = output["tokens"]
    active_tokens = output["active_patch_count"]
    compression = output["compression_ratio"]

    print(f"[Result] Total Micro-Cube Patches: 512 (4x4x4 each)")
    print(f"         Active Non-Empty Patches Tokenized: {active_tokens} tokens")
    print(f"         Pruned Pure-Air Patches: {512 - active_tokens} (saved ~{(512 - active_tokens)/512*100:.1f}% sequence budget)")
    print(f"         Voxel-to-Token Compression Ratio: {compression:.1f}x (from {total_voxels} voxels -> {active_tokens} tokens)")
    print(f"         Discrete 3D Token Indices (first 10): {tokens[0, :10].tolist()}")
    print(f"         Continuous Tokens Shape for Transformer: {output['continuous_tokens'].shape}")
    print(f"         Reconstruction Loss: {output['rec_loss'].item():.4f}")
    print(f"         Codebook VQ Loss: {output['vq_loss'].item():.4f}")
    print(f"         Total Optimization Loss: {output['total_loss'].item():.4f}")

    # Verify backward gradient flow
    print("\n[Verification] Testing backward gradient computation...")
    output["total_loss"].backward()
    print("  -> Backward pass successful! All gradients computed cleanly.")
    print("=" * 70)
    print("SphereTok is ready for training and Transformer backbone integration!")
    print("=" * 70)

if __name__ == "__main__":
    main()
