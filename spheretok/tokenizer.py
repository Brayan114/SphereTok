import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VectorQuantizer(nn.Module):
    """
    3D Vector Quantization (VQ) layer with codebook lookup and straight-through estimator.
    """
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 128, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        # Codebook weights initialized with uniform distribution
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z: torch.Tensor):
        """
        Inputs:
            z: Continuous latent vectors of shape (B, N, D)
        Returns:
            z_q: Quantized vectors of shape (B, N, D)
            vq_loss: Scalar codebook + commitment loss
            indices: Discrete token indices of shape (B, N)
        """
        # Flatten batch and sequence dimensions: (B * N, D)
        z_flat = z.reshape(-1, self.embedding_dim)

        # Compute L2 distance between latents and codebook entries: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z.e
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        # Find nearest codebook entries
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices).view(z.shape)

        # Codebook and Commitment losses
        loss_codebook = F.mse_loss(z_q, z.detach())
        loss_commitment = F.mse_loss(z_q.detach(), z)
        vq_loss = loss_codebook + self.beta * loss_commitment

        # Straight-through gradient estimator
        z_q = z + (z_q - z).detach()

        return z_q, vq_loss, indices.view(z.shape[0], z.shape[1])


class EgocentricPositionalEncoding(nn.Module):
    """
    Computes relative egocentric positional embeddings in spherical coordinates (r, theta, phi)
    relative to the agent's current position and view vector.
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(3, embedding_dim // 2),
            nn.SiLU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )

    def forward(self, patch_coords: torch.Tensor, agent_yaw: float = 0.0, agent_pitch: float = 0.0):
        """
        Inputs:
            patch_coords: Relative Cartesian coordinates (dx, dy, dz) of shape (B, N, 3)
            agent_yaw: Agent's horizontal rotation angle in radians
            agent_pitch: Agent's vertical gaze angle in radians
        Returns:
            pos_emb: Continuous positional embeddings of shape (B, N, D)
        """
        dx = patch_coords[..., 0]
        dy = patch_coords[..., 1]
        dz = patch_coords[..., 2]

        r = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-6)
        yaw_rel = torch.atan2(dz, dx) - agent_yaw
        pitch_rel = torch.asin(torch.clamp(dy / r, -1.0 + 1e-6, 1.0 - 1e-6)) - agent_pitch

        # Normalize coordinates into spherical input tensor
        spherical = torch.stack([r / 16.0, yaw_rel / math.pi, pitch_rel / (math.pi / 2.0)], dim=-1)
        return self.mlp(spherical)


class PatchEncoder3D(nn.Module):
    """
    Encodes 4x4x4 micro-cube voxel patches into continuous latent vectors.
    """
    def __init__(self, in_channels: int = 32, latent_dim: int = 128, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            nn.Conv3d(64, latent_dim, kernel_size=4, stride=2, padding=1), # (B, 128, 2, 2, 2)
            nn.BatchNorm3d(latent_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1) # (B, 128, 1, 1, 1)
        )

    def forward(self, patches: torch.Tensor):
        """
        Inputs:
            patches: Micro-cubes of shape (K, in_channels, P, P, P)
        Returns:
            latents: Continuous latents of shape (K, latent_dim)
        """
        x = self.conv(patches)
        return x.flatten(1)


class PatchDecoder3D(nn.Module):
    """
    Decodes continuous/quantized latent vectors back into 4x4x4 voxel block predictions.
    """
    def __init__(self, out_channels: int = 32, latent_dim: int = 128, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(latent_dim, 64 * 2 * 2 * 2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            nn.ConvTranspose3d(64, out_channels, kernel_size=4, stride=2, padding=1) # (K, out_channels, 4, 4, 4)
        )

    def forward(self, z_q: torch.Tensor):
        """
        Inputs:
            z_q: Latent vectors of shape (K, latent_dim)
        Returns:
            reconstruction: Voxel logits of shape (K, out_channels, 4, 4, 4)
        """
        x = self.proj(z_q).view(-1, 64, 2, 2, 2)
        return self.deconv(x)


class SphereTok(nn.Module):
    """
    SphereTok: Egocentric 3D World Tokenizer for Minecraft Agents.
    
    Transforms an egocentric (32 x 32 x 32) voxel volume into discrete 3D World Tokens.
    """
    def __init__(
        self,
        num_block_classes: int = 32,
        grid_size: int = 32,
        patch_size: int = 4,
        latent_dim: int = 128,
        codebook_size: int = 512,
        air_class_id: int = 0
    ):
        super().__init__()
        self.num_block_classes = num_block_classes
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.air_class_id = air_class_id

        self.num_patches_per_axis = grid_size // patch_size
        self.total_patches = self.num_patches_per_axis ** 3 # 8 * 8 * 8 = 512 patches

        # One-hot embedding for categorical block IDs
        self.block_embed = nn.Embedding(num_block_classes, num_block_classes)
        self.block_embed.weight.data = torch.eye(num_block_classes)
        self.block_embed.weight.requires_grad = False

        self.encoder = PatchEncoder3D(in_channels=num_block_classes, latent_dim=latent_dim, patch_size=patch_size)
        self.vq = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=latent_dim)
        self.pos_encoder = EgocentricPositionalEncoding(embedding_dim=latent_dim)
        self.decoder = PatchDecoder3D(out_channels=num_block_classes, latent_dim=latent_dim, patch_size=patch_size)

        # Precompute relative patch centers (dx, dy, dz) centered around (0, 0, 0)
        coords = torch.arange(self.num_patches_per_axis) * patch_size + (patch_size / 2.0) - (grid_size / 2.0)
        grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
        patch_centers = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1) # (512, 3)
        self.register_buffer("patch_centers", patch_centers)

    def extract_patches(self, voxel_grid: torch.Tensor):
        """
        Partitions (B, D, H, W) voxel grid into (B, 512, P, P, P) micro-cubes.
        """
        B, D, H, W = voxel_grid.shape
        P = self.patch_size
        N = self.num_patches_per_axis

        patches = voxel_grid.view(B, N, P, N, P, N, P)
        patches = patches.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        patches = patches.view(B, N * N * N, P, P, P)
        return patches

    def reconstruct_volume(self, reconstructed_patches: torch.Tensor, active_indices: torch.Tensor, batch_size: int):
        """
        Reassembles reconstructed micro-cubes back into full (B, C, D, H, W) volume.
        """
        P = self.patch_size
        N = self.num_patches_per_axis
        full_recon = torch.zeros(
            batch_size, self.total_patches, self.num_block_classes, P, P, P,
            device=reconstructed_patches.device
        )
        full_recon[:, active_indices] = reconstructed_patches
        full_recon = full_recon.view(batch_size, N, N, N, self.num_block_classes, P, P, P)
        full_recon = full_recon.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return full_recon.view(batch_size, self.num_block_classes, self.grid_size, self.grid_size, self.grid_size)

    def forward(self, voxel_grid: torch.Tensor, yaw: float = 0.0, pitch: float = 0.0, prune_air: bool = True):
        """
        Forward pass:
        1. Patchify voxel grid.
        2. Identify and prune purely air patches (sparse tokenization).
        3. 3D Encode non-empty patches.
        4. Vector Quantize to discrete 3D tokens.
        5. Inject relative egocentric positional embeddings.
        6. Decode and compute reconstruction loss.
        """
        B = voxel_grid.shape[0]
        # (B, 512, 4, 4, 4)
        patches = self.extract_patches(voxel_grid)

        # Sparse occupancy mask: does patch contain any non-air block?
        non_air_mask = (patches != self.air_class_id).any(dim=-1).any(dim=-1).any(dim=-1) # (B, 512)

        if prune_air and non_air_mask.any():
            active_idx = torch.nonzero(non_air_mask[0]).squeeze(-1)
            active_patches = patches[:, active_idx] # (B, K, 4, 4, 4)
            active_centers = self.patch_centers[active_idx].unsqueeze(0).expand(B, -1, -1)
        else:
            active_idx = torch.arange(self.total_patches, device=voxel_grid.device)
            active_patches = patches
            active_centers = self.patch_centers.unsqueeze(0).expand(B, -1, -1)

        K = active_patches.shape[1]

        # Convert block classes to one-hot channels: (B * K, C, 4, 4, 4)
        flat_active = active_patches.reshape(-1, self.patch_size, self.patch_size, self.patch_size)
        one_hot = self.block_embed(flat_active).permute(0, 4, 1, 2, 3)

        # 3D Encode: (B, K, latent_dim)
        z = self.encoder(one_hot).view(B, K, self.latent_dim)

        # Vector Quantize: (B, K, latent_dim), loss, indices (B, K)
        z_q, vq_loss, token_indices = self.vq(z)

        # Egocentric Positional Encoding
        pos_emb = self.pos_encoder(active_centers, agent_yaw=yaw, agent_pitch=pitch)
        tokens_with_pos = z_q + pos_emb

        # Decode
        recon_patches = self.decoder(z_q.view(-1, self.latent_dim))
        recon_patches = recon_patches.view(B, K, self.num_block_classes, self.patch_size, self.patch_size, self.patch_size)

        # Reconstruction loss (Cross Entropy over non-empty patches)
        rec_loss = F.cross_entropy(
            recon_patches.view(-1, self.num_block_classes),
            flat_active.view(-1)
        )

        total_loss = rec_loss + vq_loss

        return {
            "tokens": token_indices,                   # Discrete 3D World Tokens (B, K)
            "continuous_tokens": tokens_with_pos,     # Tokens with 3D egocentric pos (B, K, D)
            "active_patch_count": K,                  # Pruned token count (e.g., 78 instead of 512)
            "compression_ratio": (self.grid_size**3) / K, # Compression factor
            "total_loss": total_loss,
            "rec_loss": rec_loss,
            "vq_loss": vq_loss
        }
