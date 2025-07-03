import math
import torch
from torch.utils.data import Dataset

# -- sphere generator --
def golden_sphere_points(n: int,
                         seed: int | None = None) -> torch.Tensor:
    """
    Uniform-ish points on the unit sphere (centre 0.5).  
    Colours are random but repeatable via `seed`.
    Returns: [n,6] tensor  (xyz rgb)   all in [0,1]
    """
    if seed is not None:
        torch.manual_seed(seed)

    idx     = torch.arange(0, n, dtype=torch.float32) + 0.5
    phi     = torch.acos(1 - 2*idx/n)
    theta   = math.tau * idx * (1 + math.sqrt(5))/2

    x       = torch.cos(theta) * torch.sin(phi)
    y       = torch.sin(theta) * torch.sin(phi)
    z       = torch.cos(phi)

    xyz     = torch.stack([x, y, z], dim=1) * 0.5 + 0.5   # map to [0,1]
    rgb     = torch.rand(n, 3)                            # random colours [0,1]

    return torch.cat([xyz, rgb], dim=1)                   # [n,6]


SPHERE_POINTS = golden_sphere_points(512, seed=42)



class SphereDataset(Dataset):
    """Yields the SAME clean sphere every time; noise is added in train loop."""
    def __init__(self, n_pts=256, num_datapoints=10_000):
        self.cloud   = golden_sphere_points(n_pts, seed=42)   # [N,6]  float32
        self.num_datapoints = num_datapoints

    def __len__(self):  return self.num_datapoints

    def __getitem__(self, _):
        return self.cloud            