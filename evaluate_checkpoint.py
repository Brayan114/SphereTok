"""
Evaluation and Inspection script for the trained SphereTok checkpoint.
Computes quantitative reconstruction accuracy, IoU, and codebook utilization
across 100 diverse procedural Minecraft chunks spanning 4 biomes:
Plains, Mountain Peaks, Deep Ravines, and Subterranean Caverns.
"""

import torch
import torch.nn.functional as F
import numpy as np
from spheretok.tokenizer import SphereTok
from spheretok.dataset import ProceduralMinecraftDataset

def main():
    print("=" * 75)
    print("  SphereTok: Multi-Biome Quantitative Evaluation Across 100 Chunks")
    print("=" * 75)

    # Load model architecture
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
    print(f"[Model Loaded] Successfully restored weights from: {checkpoint_path}")

    # Generate 100 multi-biome evaluation chunks (25 per biome)
    biomes = ["plains", "mountains", "ravine", "cavern"]
    results_by_biome = {b: [] for b in biomes}
    tokens_by_biome = {b: [] for b in biomes}
    used_tokens = set()

    test_dataset = ProceduralMinecraftDataset(num_samples=100, biome="mixed", seed=9999)
    print("\n[Benchmarking] Running inference across 100 diverse multi-biome test chunks...")

    with torch.no_grad():
        for i in range(len(test_dataset)):
            chunk = test_dataset[i].unsqueeze(0) # (1, 32, 32, 32)
            biome = biomes[i % len(biomes)]

            # 1. Sparse affordance tokenization
            out_sparse = model(chunk, prune_air=True)
            tokens_by_biome[biome].append(out_sparse["active_patch_count"])

            # 2. Full 512-patch reconstruction evaluation
            out_dense = model(chunk, prune_air=False)
            tokens = out_dense["tokens"]
            used_tokens.update(tokens.flatten().tolist())

            z_q = model.vq.embedding(tokens.view(-1))
            recon_logits = model.decoder(z_q) # (512, 16, 4, 4, 4)
            pred_volume = model.reconstruct_volume(recon_logits, torch.arange(512), batch_size=1).argmax(dim=1)

            acc = (pred_volume == chunk).float().mean().item() * 100
            pred_solid = (pred_volume != 0)
            gt_solid = (chunk != 0)
            intersection = (pred_solid & gt_solid).float().sum().item()
            union = (pred_solid | gt_solid).float().sum().item()
            iou = (intersection / (union + 1e-6)) * 100

            results_by_biome[biome].append((acc, iou))

    print("\n" + "-" * 75)
    print("  Quantitative Evaluation Results by Biome (25 Chunks Each, N=100 Total):")
    print("-" * 75)

    all_accs, all_ious, all_toks = [], [], []
    for b in biomes:
        b_accs = [r[0] for r in results_by_biome[b]]
        b_ious = [r[1] for r in results_by_biome[b]]
        b_toks = tokens_by_biome[b]

        all_accs.extend(b_accs)
        all_ious.extend(b_ious)
        all_toks.extend(b_toks)

        print(f"  * {b.capitalize():11s} (N=25): Acc: {np.mean(b_accs):.2f}% +/- {np.std(b_accs):.2f}% | "
              f"IoU: {np.mean(b_ious):.2f}% +/- {np.std(b_ious):.2f}% | "
              f"Active Tokens: {np.mean(b_toks):.1f} ({32768/np.mean(b_toks):.1f}x)")

    print("-" * 75)
    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    mean_iou = np.mean(all_ious)
    std_iou = np.std(all_ious)
    mean_toks = np.mean(all_toks)
    codebook_util = (len(used_tokens) / 512) * 100

    print(f"  * OVERALL (N=100 Chunks):")
    print(f"    - Mean Voxel Reconstruction Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"    - Mean Solid Terrain Surface IoU:    {mean_iou:.2f}% +/- {std_iou:.2f}%")
    print(f"    - Mean Active Affordance Tokens:     {mean_toks:.1f} ({32768/mean_toks:.1f}x compression)")
    print(f"    - Unique Codebook Tokens Utilized:   {len(used_tokens)} / 512 ({codebook_util:.1f}%)")
    print("=" * 75)

if __name__ == "__main__":
    main()
