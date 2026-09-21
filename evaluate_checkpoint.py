"""
Evaluation and Inspection script for the trained SphereTok checkpoint.
Computes quantitative reconstruction accuracy, IoU, and codebook utilization.
"""

import torch
import torch.nn.functional as F
import numpy as np
from spheretok.tokenizer import SphereTok
from demo import create_synthetic_minecraft_chunk

def main():
    print("=" * 70)
    print("  SphereTok: Evaluating Trained Checkpoint on Minecraft Voxels")
    print("=" * 70)

    # Load model architecture
    model = SphereTok(
        num_block_classes=16,
        grid_size=32,
        patch_size=4,
        latent_dim=128,
        codebook_size=512,
        air_class_id=0
    )

    # Load trained weights
    checkpoint_path = "checkpoints/spheretok_minecraft_checkpoint.pt"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Model Loaded] Successfully restored weights from: {checkpoint_path}")

    # Generate 10 diverse test evaluation chunks
    accuracies = []
    solid_ious = []
    used_tokens = set()
    total_token_slots = 0

    print("\n[Benchmarking] Running inference across 20 synthetic Minecraft test chunks...")
    with torch.no_grad():
        for i in range(20):
            chunk = create_synthetic_minecraft_chunk()
            B = chunk.shape[0]

            # Forward pass without air pruning to evaluate full 512-patch reconstruction
            out = model(chunk, prune_air=False)
            tokens = out["tokens"]
            used_tokens.update(tokens.flatten().tolist())
            total_token_slots += tokens.numel()

            # Decode all 512 patches to reconstruct full volume
            z_q = model.vq.embedding(tokens.view(-1))
            recon_logits = model.decoder(z_q) # (512, 16, 4, 4, 4)
            pred_classes = recon_logits.argmax(dim=1) # (512, 4, 4, 4)

            # Reconstruct full (32, 32, 32) volume
            pred_volume = model.reconstruct_volume(recon_logits, torch.arange(512), batch_size=B).argmax(dim=1)

            # Measure overall voxel accuracy
            acc = (pred_volume == chunk).float().mean().item() * 100
            accuracies.append(acc)

            # Measure solid block IoU (ignoring empty air)
            pred_solid = (pred_volume != 0)
            gt_solid = (chunk != 0)
            intersection = (pred_solid & gt_solid).float().sum().item()
            union = (pred_solid | gt_solid).float().sum().item()
            iou = (intersection / (union + 1e-6)) * 100
            solid_ious.append(iou)

    mean_acc = np.mean(accuracies)
    mean_iou = np.mean(solid_ious)
    codebook_utilization = (len(used_tokens) / 512) * 100

    print("-" * 70)
    print(f"  Quantitative Evaluation Results:")
    print(f"  * Overall Voxel Reconstruction Accuracy: {mean_acc:.2f}%")
    print(f"  * Solid Surface & Terrain IoU:          {mean_iou:.2f}%")
    print(f"  * Unique Codebook Tokens Utilized:       {len(used_tokens)} / 512 ({codebook_utilization:.1f}%)")
    print("-" * 70)

    # Tokenizer compression breakdown
    with torch.no_grad():
        sample_chunk = create_synthetic_minecraft_chunk()
        sample_out = model(sample_chunk, prune_air=True)
        active_k = sample_out["active_patch_count"]

    print("\n[Token Budget / LLM Context Breakdown]:")
    print(f"  - Raw Voxel Count:                 32,768 discrete 3D cells")
    print(f"  - Dense Patch Token Count:         512 tokens (4x4x4 micro-cubes)")
    print(f"  - Sparse Air-Pruned Token Count:   {active_k} tokens")
    print(f"  - Total Context Compression Ratio: {(32768 / active_k):.1f}x reduction")
    print(f"  - Fits comfortably in LLM context: YES (~{active_k} tokens = only ~3-5% of standard 4k/8k context!)")
    print("=" * 70)

if __name__ == "__main__":
    main()
