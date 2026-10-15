"""
Procedural Multi-Biome Minecraft Terrain Dataset for SphereTok.
Synthesizes diverse 32x32x32 voxel chunks across 4 distinct Minecraft environmental biomes:
1. Plains: Rolling gentle hills with soil, trees, and small underground cavities.
2. Mountains: High vertical elevation peaks (Y up to 28), sheer slopes, and exposed rock.
3. Ravines: Deep vertical canyon chasms cutting through bedrock strata with steep walls.
4. Caverns: Dense subterranean cave chambers with interconnected winding passageways and water pools.
"""

import torch
from torch.utils.data import Dataset
import numpy as np

class ProceduralMinecraftDataset(Dataset):
    """
    Synthesizes diverse procedural 3D Minecraft chunks across multiple biomes:
    - 0: Air
    - 1: Stone (Subterranean strata, canyon walls, mountain rock)
    - 2: Grass / Soil (Surface terrain layer)
    - 3: Wood (Tree trunks / obstacle pillars)
    - 4: Water (Subterranean aquifers / surface pools)
    """
    def __init__(self, num_samples: int = 1000, grid_size: int = 32, biome: str = "mixed", seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.biome = biome
        self.seed = seed
        self.biomes_list = ["plains", "mountains", "ravine", "cavern"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        rng = np.random.RandomState(self.seed + idx)
        grid = torch.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=torch.long)

        # Select biome
        if self.biome == "mixed":
            active_biome = self.biomes_list[idx % len(self.biomes_list)]
        else:
            active_biome = self.biome

        if active_biome == "plains":
            base_h = rng.randint(8, 12)
            x_c = np.linspace(0, 2 * np.pi, self.grid_size)
            z_c = np.linspace(0, 2 * np.pi, self.grid_size)
            X, Z = np.meshgrid(x_c, z_c)
            terrain_h = np.clip(
                (base_h + 2.0 * np.sin(X + rng.uniform(0, 3)) + 1.5 * np.cos(Z + rng.uniform(0, 3))).astype(int),
                4, 16
            )
            for x in range(self.grid_size):
                for z in range(self.grid_size):
                    h = terrain_h[x, z]
                    grid[0:max(1, h - 2), x, z] = 1 # Stone
                    grid[max(0, h - 2):h, x, z] = 2 # Grass / Soil

            # Tree / obstacle pillar
            if rng.rand() > 0.3:
                px, pz = rng.randint(6, 26, size=2)
                ph = rng.randint(4, 7)
                gy = terrain_h[px, pz]
                grid[gy:min(self.grid_size, gy + ph), px, pz] = 3

            # Underground cave
            if rng.rand() > 0.4:
                cx, cy, cz = rng.randint(8, 24, size=3)
                cr = rng.randint(3, 5)
                for y in range(max(1, cy - cr), min(self.grid_size, cy + cr)):
                    for x in range(max(0, cx - cr), min(self.grid_size, cx + cr)):
                        for z in range(max(0, cz - cr), min(self.grid_size, cz + cr)):
                            if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= cr**2:
                                grid[y, x, z] = 0

        elif active_biome == "mountains":
            base_h = rng.randint(14, 18)
            x_c = np.linspace(0, 3 * np.pi, self.grid_size)
            z_c = np.linspace(0, 3 * np.pi, self.grid_size)
            X, Z = np.meshgrid(x_c, z_c)
            terrain_h = np.clip(
                (base_h + 6.0 * np.sin(X + rng.uniform(0, 3)) + 4.0 * np.cos(Z + rng.uniform(0, 3))).astype(int),
                6, 28
            )
            for x in range(self.grid_size):
                for z in range(self.grid_size):
                    h = terrain_h[x, z]
                    grid[0:max(1, h - 1), x, z] = 1 # Mountain rock
                    grid[max(0, h - 1):h, x, z] = 2 # Peak soil / snow

        elif active_biome == "ravine":
            grid[0:4, :, :] = 1 # Canyon bedrock floor
            rx = rng.randint(12, 20)
            rw = rng.randint(4, 7)
            for x in range(self.grid_size):
                for z in range(self.grid_size):
                    offset = int(2.5 * np.sin(z / 4.0))
                    if abs(x - (rx + offset)) > (rw // 2):
                        wh = rng.randint(18, 26)
                        grid[4:wh, x, z] = 1
                        if wh > 21:
                            grid[wh - 1:wh, x, z] = 2

        elif active_biome == "cavern":
            grid[0:28, :, :] = 1 # Solid subterranean bedrock
            for _ in range(rng.randint(2, 4)):
                cx, cy, cz = rng.randint(8, 24, size=3)
                cr = rng.randint(4, 7)
                for y in range(max(1, cy - cr), min(31, cy + cr)):
                    for x in range(max(1, cx - cr), min(31, cx + cr)):
                        for z in range(max(1, cz - cr), min(31, cz + cr)):
                            if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= cr**2:
                                grid[y, x, z] = 0 # Air cavity
            # Subterranean water pool
            grid[1:3, 10:22, 10:22] = 4

        return grid
