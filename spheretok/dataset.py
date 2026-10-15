"""
Procedural Minecraft Terrain Dataset for SphereTok.
Generates realistic 32x32x32 voxel chunks featuring undulating terrain,
subterranean cave networks, and surface obstacle pillars.
"""

import torch
from torch.utils.data import Dataset
import numpy as np

class ProceduralMinecraftDataset(Dataset):
    """
    Synthesizes diverse procedural 3D Minecraft chunks:
    - 0: Air
    - 1: Stone (bedrock and underground subterranean strata)
    - 2: Soil / Grass (surface terrain envelope)
    - 3: Wood (vertical obstacle pillars / trees)
    - 4: Water
    """
    def __init__(self, num_samples: int = 1000, grid_size: int = 32, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # Deterministic per-sample seed for reproducibility
        rng = np.random.RandomState(self.seed + idx)

        grid = torch.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=torch.long)

        # Baseline ground elevation (Y from 8 to 14)
        base_h = rng.randint(8, 14)

        # Generate undulating terrain using composite sinusoidal waves
        x_coords = np.linspace(0, 2 * np.pi, self.grid_size)
        z_coords = np.linspace(0, 2 * np.pi, self.grid_size)
        X, Z = np.meshgrid(x_coords, z_coords)
        terrain_height = (
            base_h
            + 3.0 * np.sin(X + rng.uniform(0, 3))
            + 2.0 * np.cos(Z + rng.uniform(0, 3))
        ).astype(int)
        terrain_height = np.clip(terrain_height, 4, 22)

        for x in range(self.grid_size):
            for z in range(self.grid_size):
                h = terrain_height[x, z]
                # Subterranean stone
                grid[0:max(1, h - 2), x, z] = 1
                # Surface soil / grass layer
                grid[max(0, h - 2):h, x, z] = 2

        # Carve random underground cave cavities (70% probability)
        if rng.rand() > 0.3:
            cave_x, cave_y, cave_z = rng.randint(8, 24, size=3)
            cave_r = rng.randint(3, 6)
            for y in range(max(1, cave_y - cave_r), min(self.grid_size, cave_y + cave_r)):
                for x in range(max(0, cave_x - cave_r), min(self.grid_size, cave_x + cave_r)):
                    for z in range(max(0, cave_z - cave_r), min(self.grid_size, cave_z + cave_r)):
                        if (x - cave_x)**2 + (y - cave_y)**2 + (z - cave_z)**2 <= cave_r**2:
                            grid[y, x, z] = 0 # Air inside cave

        # Add vertical surface obstacle columns / tree trunks (60% probability)
        if rng.rand() > 0.4:
            px, pz = rng.randint(6, 26, size=2)
            pillar_h = rng.randint(3, 7)
            ground_y = terrain_height[px, pz]
            grid[ground_y:min(self.grid_size, ground_y + pillar_h), px, pz] = 3

        return grid
