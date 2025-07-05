from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from   torch.utils.data import DataLoader

from dataset import SphereDataset
from itertools import pairwise


class T_Embed(nn.Module):
    def __init__(self, out_dim: int = 4):
        super().__init__() ; self.mlp = nn.Sequential(nn.Linear(1, 32), nn.ReLU(),
                                                      nn.Linear(32, out_dim))

    def forward(self, t): return self.mlp(t)

class GenerativeExpert_Axis(nn.Module):

    MASK = {'x': 0, 'y': 1, 'z': 2, 'all_axes': slice(0, 3), 'all_colours': slice(3, 6)}

    def __init__(self,
                 axis: str,
                 hidden: list[int] = [256, 256, 256],
                 lr: float = 3e-4,
                 n_epochs: int = 500,
                 t_dim: int = 4,
                 device: str = 'cpu'):
        super().__init__()
        assert axis in (self.MASK.keys() - ['all_colours'])
        self.axis      = axis
        self.axis_i    = self.MASK[axis]
        self.mask      = torch.zeros(6, dtype=torch.bool)


        self.mask[self.axis_i] = True ; self.mask[self.MASK['all_colours']] = False

        self.t_embed    = T_Embed(t_dim).to(device)
        self.t_dim      = t_dim

        dims   = [6+self.t_dim, *hidden, 6] # in= 6 coord+1 time
        layers = []

        for idx, (d_in, d_out) in enumerate(pairwise(dims)):
            layers                          += [nn.Linear(d_in, d_out)]
            if idx < len(dims) - 2: layers  += [nn.ReLU()] # -- add relu unless it's the last layer
        
        self.net    = nn.Sequential(*layers).to(device)
        self.opt    = optim.Adam(self.parameters(), lr=lr)
        self.epochs = n_epochs
        self.sched  = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=n_epochs)
        self.device = device

    @property
    def name(self):
        axis_str = 'a' if self.axis == 'all_axes' else self.axis
        return f"expert_{axis_str}"

    def calculate_velocity(self, pts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        pts : [B,N,6]  cloud batch
        t   : [B,1] or scalar
        Returns masked velocity [B,N,6] (zeros on unsupervised slots).
        """
        B, N, _     = pts.shape
        t           = self.t_embed(t.expand(B, N, 1))
        inp         = torch.cat([pts, t], dim=-1)   # [B,N,7+self.t_dim]
        out         = self.net(inp.view(-1, 6 + self.t_dim))

        # -- apply mask based on which expert we are
        v_axis      = out[:, self.axis_i]

        # -- create new tensor so shapes match up (output has out_dim shape, not 6)
        vel                 = torch.zeros_like(pts.reshape(-1, 6))
        vel[:, self.axis_i] = v_axis

        return vel.view(B, N, 6)

    # ─────────────────────────────────────────  training  ──
    def train_loop(self,
                   loader: DataLoader,
                   ckpt_dir: str | Path | None = None):
        """
        Flow-matching objective for linear (rectified-flow) forward process:
            x_t = (1-t)·x₀ + t·ε        ⇒   v* = ε - x₀
        We sample ε once per batch; t ~ U(0,1).
        Loss is MSE on the **masked** dimensions only.
        """

        mask        = self.mask.to(self.device)            # [6]
        mse         = nn.MSELoss()

        for ep in tqdm(range(1, self.epochs + 1)):
            tot, n = 0.0, 0
            for cloud in loader:                    # cloud [B,N,6]
                cloud   = cloud.to(self.device)
                B, N, _ = cloud.shape

                eps     = torch.rand_like(cloud)                 # ε
                t       = torch.rand(B, 1, 1, device=self.device)
                x_t     = (1 - t) * cloud + t * eps              # noisy

                pred    = self.calculate_velocity(x_t, t)

                # -- for this toy example, model can only optimize for its selected expertise
                target  = (eps - cloud)[:, :, mask]
                pred    = pred         [:, :, mask]

                loss    = mse(pred, target)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                tot += loss.item(); n += 1
            
            print(f"epoch {ep:02d}  loss {tot/n:.6f}")
            tot = 0.0; n = 0

        # -- models are so small so just save the finished model
        if ckpt_dir:
            Path(ckpt_dir).mkdir(exist_ok=True, parents=True)
            torch.save(self.state_dict(), Path(ckpt_dir)/f"{self.name}.pt")


class GenerativeExpert_Color(nn.Module):

    CH_IDX = {'r': 3, 'g': 4, 'b': 5}        # xyz(0-2)  rgb(3-5)

    def __init__(self, color: str, device: str = 'cpu'):
        super().__init__()
        self.color = color
        self.device = device
    
    def forward(self, cloud: torch.Tensor) -> torch.Tensor:
        pass

    def train_loop(self, loader: DataLoader, ckpt_dir: str | Path | None = None):
        pass


class DiscriminativeExpert_Color(nn.Module):
    """
    Scalar reward expert: high reward when points are predominantly `color`.
    Returns one log-reward per particle (≥0 good, ≤0 bad).
    """
    CH_IDX = {'r': 3, 'g': 4, 'b': 5}        # xyz(0-2)  rgb(3-5)

    def __init__(self, color: str, coef: float = 20.0, device: str = 'cpu'):
        super().__init__()
        assert color in self.CH_IDX or color == 'white'
        self.color   = color
        self.coef    = coef                  # sharpness λ
        self.device  = device

    @torch.no_grad()
    def score(self, cloud: torch.Tensor) -> torch.Tensor:
        """
        cloud : [L,N,6] or [N,6]
        Returns : log-reward  (shape [L] or scalar)
        """
        # promote [N,6] → [1,N,6]
        if cloud.ndim == 2: cloud = cloud.unsqueeze(0)

        rgb   = cloud[..., 3:6]
        # reward high overall brightness for all-colors
        if self.color == 'white': reward = rgb.mean(dim=(-2, -1))
        else:
            idx     = self.CH_IDX[self.color] - 3
            others  = rgb[:, [i for i in range(3) if i != idx]].mean(-1)
            reward  = rgb[:, idx].mean(-1) - others.mean(-1)

        return self.coef * reward

    def train_loop(self, loader: DataLoader, ckpt_dir: str | Path | None = None):
        pass


# ───────────────────────────────────── demo run ──
if __name__ == "__main__":
    ds      = SphereDataset(num_datapoints=10_000)
    loader  = DataLoader(ds, batch_size=64, shuffle=True)   # whole cloud per batch
    CHECKPOINT_DIR = Path(__file__).parent / 'checkpoints'

    expert_xr  = GenerativeExpert_Axis(axis='x', device='cuda:1', n_epochs=100)
    expert_yg  = GenerativeExpert_Axis(axis='y', device='cuda:1', n_epochs=100)
    expert_zb  = GenerativeExpert_Axis(axis='z', device='cuda:1', n_epochs=100)
    expert_all = GenerativeExpert_Axis(axis='all_axes', device='cuda:1', n_epochs=100)
    
    expert_xr .train_loop(loader, ckpt_dir=CHECKPOINT_DIR)
    expert_yg .train_loop(loader, ckpt_dir=CHECKPOINT_DIR)
    expert_zb .train_loop(loader, ckpt_dir=CHECKPOINT_DIR)
    expert_all.train_loop(loader, ckpt_dir=CHECKPOINT_DIR)

    expert_xr .load_state_dict(torch.load(CHECKPOINT_DIR / 'expert_x.pt'))
    expert_yg .load_state_dict(torch.load(CHECKPOINT_DIR / 'expert_y.pt'))
    expert_zb .load_state_dict(torch.load(CHECKPOINT_DIR / 'expert_z.pt'))
    expert_all.load_state_dict(torch.load(CHECKPOINT_DIR / 'expert_all.pt'))

    noise = torch.randn(1000, 6, device='cuda:1')
