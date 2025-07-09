import  math
import  torch
from    torch import Tensor
from    torch.utils.data import Dataset
from    functools import cache

@cache
def get_palette(device: str | None = None) -> Tensor:
    return torch.tensor(
        [
            [0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.],
            [0.,0.,1.],[1.,0.,1.],[0.,1.,1.],[1.,1.,1.]
        ],
        dtype=torch.float32
    ).to(device or torch.get_default_device())


def golden_sphere_points(n: int, seed: int | None = None) -> Tensor:
    """
    Uniform-ish xyz points on a unit sphere, re-centred to [0, 1]^3.
    Shape: [n, 3]  (xyz)
    """
    if seed is not None:
        torch.manual_seed(seed)

    idx   = torch.arange(0, n, dtype=torch.float32) + 0.5
    phi   = torch.acos(1 - 2 * idx / n)
    theta = math.tau * idx * (1 + math.sqrt(5)) / 2

    xyz   = torch.stack([
        torch.cos(theta) * torch.sin(phi),   # x
        torch.sin(theta) * torch.sin(phi),   # y
        torch.cos(phi)                       # z
    ], dim=1)

    return xyz * 0.5 + 0.5                  # map from [-1,1] -> [0,1]


def _sample_rgb(n_pts:      int,
                n_modes:    int = 3,
                sigma:      float = 0.10,
                seed:       int | None = None) -> Tensor:
    """
    Draw colours from a K-component mixture of palette corners.
    Returns [n_pts, 3]
    """
    if seed is not None:
        torch.manual_seed(seed)
    palette  = get_palette()
    idx      = torch.randperm(len(palette), device=palette.device)[:n_modes]         # pick K distinct centres
    centres  = palette[idx]                                   # [K, 3]
    assigns  = torch.arange(n_pts) % n_modes                  # round-robin assignment
    rgb      = centres[assigns] + sigma * torch.randn(n_pts, 3)

    return rgb.clamp(0, 1)

def sample_noise(n_pts: int,
                 sigma: float      = 0.40,
                 seed:  int | None = None,
                 device: str | None = None) -> Tensor:
    """
    Samples a sphere with an rgb profile centered around a random color in the palette.
    Returns [n_pts, 6]
    """
    return torch.cat([
        torch.rand_like(golden_sphere_points(n_pts, seed)),
        _sample_rgb(n_pts, sigma=sigma, seed=seed)
    ], dim=1).to(device or torch.get_default_device())


class SphereDataset(Dataset):
    """
    Each item: clean static geometry + per-sample colour mixture noise.
    Output shape: [n_pts, 6]  (xyz | rgb) on the dataset's device.
    """
    def __init__(
        self,
        n_pts:      int     = 256,
        n_samples:  int     = 10_000,
        sigma:      float   = 0.10,
        device:     str     = 'cuda:0',
        seed:       int     = 24
    ):
        self.n_pts     = n_pts
        self.n_samples = n_samples
        self.sigma     = sigma

        self.device = device or torch.get_default_device()

        # pre-compute the static xyz cloud once and move to device
        self.cloud_xyz = golden_sphere_points(n_pts, seed=seed)
        super().__init__()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, _: int) -> Tensor:
        # colours are generated fresh each call, then moved to the dataset device
        rgb = _sample_rgb(self.n_pts, sigma=self.sigma)
        return torch.cat([self.cloud_xyz, rgb], dim=1)

if __name__ == '__main__':
    dataset = SphereDataset(n_pts=1024, n_samples=10000, sigma=0.10, device='cuda')
    print(dataset[0].shape)
    print(dataset[0])