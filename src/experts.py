from torch.types import Number
from tqdm.auto import tqdm
from pathlib import Path
from itertools import pairwise

import  torch
from    torch import Tensor
import  torch.nn as nn
import  torch.optim as optim
from    torch import Tensor
from    torch.utils.data import DataLoader

from src.dataset import sample_noise


class T_Embed(nn.Module):
    def __init__(self, out_dim: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, out_dim))

    def forward(self, t): return self.mlp(t)


class BaseFlowExpert(nn.Module):

    IDX = {'x': 0, 'y': 1, 'z': 2,
           'r': 0, 'g': 1, 'b': 2,
           'all_axes':  slice(0,3),
           'rgb':       slice(0,3)}

    def __init__(self,
                 mask: Tensor,
                 t_dim: int = 4,
                 in_dim: int | None = None,
                 out_dim: int = 6,
                 lr: float = 3e-4,
                 n_epochs: int = 500,
                 device: str | None = None):

        super().__init__()

        self.device = device or torch.get_default_device()
        self.mask   = mask.to(self.device)
        self.t_dim  = t_dim
        self.in_dim = in_dim or out_dim
        self.out_dim = out_dim
        self.epochs = n_epochs

        self.t_embed = T_Embed(t_dim).to(self.device)

        hidden = [256,256,256]
        dims   = [self.in_dim+t_dim, *hidden, out_dim] # always output 6 columns
        layers = []

        for i,(d_in,d_out) in enumerate(pairwise(dims)):
            layers                     += [nn.Linear(d_in,d_out)]
            if i < len(dims)-2: layers += [nn.ReLU()]

        self.net   = nn.Sequential(*layers).to(self.device)

        self.opt   = optim.Adam(self.parameters(), lr=lr)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=n_epochs)

    @property
    def name(self) -> str: return 'base_expert'

    def calculate_velocity(self, pts: Tensor, t: Number) -> Tensor:
        B, N, _  = pts.shape
        t_emb    = self.t_embed(torch.tensor(t).expand(B,N,1))
        pts_feat = torch.cat([pts,t_emb],dim=-1)
        # -- run the net on the flattened features
        out      = self.net(pts_feat.view(-1, pts_feat.shape[-1]))

        # -- mask is zero (discarded) by default
        vel                 = torch.zeros_like(out) 
        vel[:, self.mask]   = out[:, self.mask]
        return              vel.view(B,N,self.out_dim)

    def calculate_target(self, cloud: Tensor, eps: Tensor) -> Tensor:
        return eps-cloud

    def train_loop(self, loader: DataLoader, ckpt_dir: Path | None = None, progress_bar: tqdm | None = None):
        self.train()
        
        mse = nn.MSELoss()

        for ep in range(1,self.epochs+1):
            running = 0; n = 0

            for cloud in loader:
                cloud   = cloud.to(self.device)
                B, N, _ = cloud.shape

                eps     = sample_noise(N, seed=ep)
                t       = torch.rand(B,1,1)
                x_t     = ((1 - t) * cloud) + (t * eps)

                # -- since the velocity is masked within `calculate_velocity`
                # but returned as [..., 6]
                pred    = self.calculate_velocity(x_t,t)
                target  = self.calculate_target  (cloud, eps)
                loss    = mse(pred, target)

                self.opt.zero_grad(); loss.backward(); self.opt.step()

                running += loss.item(); n += 1
                if progress_bar: progress_bar.set_description(f"Epoch {ep:03d} - Loss {running/n:.4f}")
            self.sched.step()
            if progress_bar: progress_bar.update(1)


        if ckpt_dir:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), ckpt_dir/(self.name+".pt"))


class GenerativeExpert_Axis(BaseFlowExpert):
    def __init__(self, axis: str, **kwargs):
        assert axis in {'x', 'y', 'z', 'all_axes'}
        mask                    = torch.zeros(3, dtype=torch.bool)
        mask[self.IDX[axis]]    = True           # velocities only on that axis
        super().__init__(mask=mask, out_dim=3, **kwargs)
        self.axis = axis

    @property
    def name(self) -> str: return f'expert_{self.axis}' if self.axis != 'all_axes' else 'expert_a'

    def calculate_velocity(self, pts: Tensor, t: Number) -> Tensor:
        # -- remove color so we don't learn with colors in mind
        # -- this helps because once we have a color expert, it will 
        # correlate colors with each other, making them OOD for an axis
        # expert with random colors
        pts = pts.clone()
        clr = pts[:, :, self.IDX['rgb']]
        axs = pts[:, :, self.IDX['all_axes']]
        vel = super().calculate_velocity(axs, t)
        return torch.cat([vel, torch.zeros_like(clr)], dim=-1)


class AnalyticExpert_Monochrome(nn.Module):
    def __init__(self, *args, scale=2.0, **kwargs):
        super().__init__()
        self.scale  = scale          # optional (1-t) or ε multiplier

    @property
    def name(self): return "expert_mono_analytic"

    @torch.no_grad()
    def calculate_velocity(self, pts: Tensor, t: Number) -> Tensor:
        rgb       = pts[..., 3:6]                              # [B,N,3]
        # -- we want to move our random noise's colors to the mean
        mean_rgb  = rgb.mean(dim=-2, keepdim=True) # [B,1,3]
        v_rgb     = (rgb - mean_rgb) * self.scale              # [B,N,3]
        zeros_xyz = torch.zeros_like(pts[..., :3])             # [B,N,3]
        return torch.cat([zeros_xyz, v_rgb], dim=-1)           # [B,N,6]

    def train_loop(self, loader: DataLoader, ckpt_dir: Path | None = None, progress_bar: tqdm | None = None):
        pass

# -- discriminative expert that gives a reward for a certain color
class DiscriminativeExpert_Color(nn.Module):
    """
    Scalar reward expert: high reward when points are predominantly `color`.
    Returns one log-reward per particle (≥0 good, ≤0 bad).
    """
    CH_IDX = {'r': 3, 'g': 4, 'b': 5}        # xyz(0-2)  rgb(3-5)

    def __init__(self, color: str, coef: float = 300.0, device: str | None = None):
        super().__init__()
        assert color in self.CH_IDX or color == 'white'
        self.color   = color
        self.coef    = coef
        self.device  = device or torch.get_default_device()

    @property
    def name(self) -> str: return f'discriminative_{self.color}'

    @torch.no_grad()
    def score(self, cloud: Tensor) -> Tensor:
        """
        cloud : [L,N,6] or [N,6]
        Returns : log-reward  (shape [L] or scalar)
        """
        # promote [N,6] -> [1,N,6]
        if cloud.ndim == 2: cloud = cloud.unsqueeze(0)

        rgb         = cloud[..., 3:6]
        rgb_mean    = rgb.mean(dim=-2)
        # reward high overall brightness for all-colors
        if self.color == 'white': reward = rgb_mean.mean(dim=-1)
        else:
            idx      = self.CH_IDX[self.color] - 3
            selected = rgb_mean[:, idx]
            others   = (rgb_mean.sum(dim=-1) - selected) / 2
            reward   = selected - others

        return self.coef * reward

    def train_loop(self, loader: DataLoader, ckpt_dir: str | Path | None = None, progress_bar: tqdm | None = None):
        pass

if __name__ == '__main__':
    from src.dataset import SphereDataset

    expert_x = GenerativeExpert_Axis(axis='x', device='cuda:0')
    expert_y = GenerativeExpert_Axis(axis='y', device='cuda:0')
    expert_z = GenerativeExpert_Axis(axis='z', device='cuda:0')
    torch.set_default_device('cuda:0')
    LOADER = DataLoader(SphereDataset(n_pts=512, device='cuda:0'), batch_size=64, shuffle=True, generator=torch.Generator(device='cuda:0'))
    expert_x.train_loop(LOADER, ckpt_dir=Path('checkpoints'), progress_bar=tqdm(range(500), desc='Training expert_x'))
    expert_y.train_loop(LOADER, ckpt_dir=Path('checkpoints'), progress_bar=tqdm(range(500), desc='Training expert_y'))
    expert_z.train_loop(LOADER, ckpt_dir=Path('checkpoints'), progress_bar=tqdm(range(500), desc='Training expert_z'))
