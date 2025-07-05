from tqdm import tqdm
from pathlib import Path

import  torch
import  torch.nn as nn
import  torch.optim as optim
from    torch import Tensor
from    torch.utils.data import DataLoader

from dataset import SphereDataset
from itertools import pairwise


class T_Embed(nn.Module):
    def __init__(self, out_dim: int = 4):
        super().__init__() ; self.mlp = nn.Sequential(nn.Linear(1, 32), nn.ReLU(),
                                                      nn.Linear(32, out_dim))

    def forward(self, t): return self.mlp(t)


class BaseFlowExpert(nn.Module):

    IDX = {'x': 0, 'y': 1, 'z': 2,
           'r': 3, 'g': 4, 'b': 5,
           'all_axes': slice(0,3),
           'rgb': slice(3,6)}

    def __init__(self,
                 mask: torch.BoolTensor,
                 hidden: list[int] = [256,256,256],
                 t_dim: int = 4,
                 out_dim: int = 6,
                 lr: float = 3e-4,
                 n_epochs: int = 500,
                 device: str = "cpu"):

        super().__init__()

        self.mask   = mask.to(device)
        self.t_dim  = t_dim
        self.out_dim = out_dim
        self.device = device
        self.epochs = n_epochs

        self.t_embed = T_Embed(t_dim).to(device)

        dims  = [out_dim+t_dim, *hidden, out_dim]          # always output 6 columns
        layers= []

        for i,(d_in,d_out) in enumerate(pairwise(dims)):
            layers                     += [nn.Linear(d_in,d_out)]
            if i < len(dims)-2: layers += [nn.ReLU()]

        self.net   = nn.Sequential(*layers).to(device)
        self.opt   = optim.Adam(self.parameters(), lr=lr)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt,T_max=n_epochs)

    @property
    def name(self) -> str:
        return 'base_expert'

    def calculate_velocity(self, pts: Tensor, t: Tensor) -> Tensor:
        B, N, _ = pts.shape
        t_emb   = self.t_embed(t.expand(B,N,1))
        out     = self.net(torch.cat([pts,t_emb],dim=-1).view(-1, self.out_dim+self.t_dim))

        # -- mask is zero (discarded) by default
        vel                 = torch.zeros_like(out)
        vel[:, self.mask]   = out[:, self.mask]
        return              vel.view(B,N,self.out_dim)

    def train_loop(self, loader: DataLoader, ckpt_dir: Path | None = None):
        self.train()
        
        mse = nn.MSELoss()

        for ep in tqdm(range(1,self.epochs+1)):
            running = 0; n = 0

            for cloud in loader:
                cloud   = cloud.to(self.device)
                B, N, _ = cloud.shape

                eps     = torch.rand_like(cloud)
                t       = torch.rand(B,1,1,device=self.device)
                x_t     = ((1 - t) * cloud) + (t * eps)

                pred    = self.calculate_velocity(x_t,t)

                # -- for this toy example, model can only optimize for its selected expertise
                target  = (eps-cloud)[:, :, self.mask]
                pred    = pred       [:, :, self.mask]

                loss    = mse(pred, target)

                self.opt.zero_grad(); loss.backward(); self.opt.step()

                running += loss.item(); n += 1
            print(f"epoch {ep:03d}  loss {running/n:.4f}")
            self.sched.step()

        if ckpt_dir:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), ckpt_dir/(self.name+".pt"))


# -- generative experts 
class GenerativeExpert_Axis(BaseFlowExpert):

    
    def __init__(self, axis: str, **kwargs):
        assert axis in {'x', 'y', 'z', 'all_axes'}
        mask                    = torch.zeros(3, dtype=torch.bool)
        mask[self.IDX[axis]]    = True           # velocities only on that axis
        super().__init__(mask=mask, out_dim=3, **kwargs)
        self.axis = axis

    @property
    def name(self) -> str: return f'expert_{self.axis}' if self.axis != 'all_axes' else 'expert_a'

    def calculate_velocity(self, pts: Tensor, t: Tensor) -> Tensor:
        # -- remove color so we don't learn with colors in mind
        # -- this helps because once we have a color expert, it will 
        # correlate colors with each other, making them OOD for an axis
        # expert with random colors
        pts = pts.clone()
        clr = pts[:, :, self.IDX['rgb']]
        pts = pts[:, :, self.IDX['all_axes']]
        vel = super().calculate_velocity(pts, t)
        return torch.cat([vel, torch.zeros_like(clr)], dim=-1)


class GenerativeExpert_Color(BaseFlowExpert):
    CH_IDX = {'r': 3, 'g': 4, 'b': 5,
              'rgb': slice(3,6)}

    def __init__(self, color: str, **kwargs):
        assert color in self.CH_IDX

        self.color                 = color
        mask                       = torch.zeros(3,dtype=torch.bool)
        mask[self.CH_IDX[color]]   = True       # velocities only on RGB channel(s)
        super().__init__(mask=mask, out_dim=3, **kwargs)

    @property
    def name(self) -> str: return f'expert_{self.color}'

    def calculate_velocity(self, pts: Tensor, t: Tensor) -> Tensor:
        pts = pts.clone()
        axs = pts[:, :, self.IDX['all_axes']]
        clr = pts[:, :, self.CH_IDX['rgb']]
        vel = super().calculate_velocity(clr, t)
        return torch.cat([torch.zeros_like(axs), vel], dim=-1)


# -- discriminative expert that gives a reward for a certain color
class DiscriminativeExpert_Color(nn.Module):
    """
    Scalar reward expert: high reward when points are predominantly `color`.
    Returns one log-reward per particle (≥0 good, ≤0 bad).
    """
    CH_IDX = {'r': 3, 'g': 4, 'b': 5}        # xyz(0-2)  rgb(3-5)

    def __init__(self, color: str, coef: float = 1.0, device: str = 'cpu'):
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

    expert_xr .load_state_dict(torch.load(CHECKPOINT_DIR / 'expert_xo.pt'))
    expert_yg .load_state_dict(torch.load(CHECKPOINT_DIR / 'expert_yo.pt'))
    expert_zb .load_state_dict(torch.load(CHECKPOINT_DIR / 'expert_zo.pt'))
    expert_all.load_state_dict(torch.load(CHECKPOINT_DIR / 'expert_allo.pt'))

    noise = torch.randn(1000, 6, device='cuda:1')

