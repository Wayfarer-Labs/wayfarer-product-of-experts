from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from   torch.utils.data import DataLoader

from dataset import SphereDataset
from itertools import pairwise

class PointWiseTransformer(nn.Module):
    def __init__(self, d_model=32, n_heads=2, depth=2, out_dim=2):
        super().__init__()
        self.proj_in  = nn.Linear(7, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
                            d_model=d_model,
                            nhead=n_heads,
                            dim_feedforward=d_model*4,
                            batch_first=True,
                            activation='gelu')
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.proj_out = nn.Linear(d_model, out_dim)

    def forward(self, pts_t):           # [B,N,7]
        h   = self.proj_in(pts_t)       # [B,N,D]
        h   = self.encoder(h)           # [B,N,D]
        out = self.proj_out(h)          # [B,N,2]
        return out


class LearnedT(nn.Module):
    def __init__(self, out_dim: int = 4):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, out_dim))
    def forward(self, t):           # t [...,1]
        return self.mlp(t)


class Expert_FlowMatching(nn.Module):

    AXIS = {'x': 0, 'y': 1, 'z': 2, 'r': 3, 'g': 4, 'b': 5,
            'all_axes':    slice(0, 3),
            'all_colours': slice(3, 6)}

    def __init__(self,
                 axis: str, colour: str,
                 hidden: list[int] = [256, 256, 256],
                 lr: float = 3e-4, device: str = 'cpu'):
        super().__init__()
        assert (axis in 'xyz' or axis == 'all_axes') and (colour in 'rgb' or colour == 'all_colours')
        self.axis      = axis
        self.colour    = colour
        self.axis_i    = self.AXIS[axis]
        self.col_i     = self.AXIS[colour]
        self.mask      = torch.zeros(6, dtype=torch.bool)
        self.mask[self.axis_i] = self.mask[self.col_i] = True

        self.t_embed    = LearnedT().to(device)
        self.t_dim      = self.t_embed.out_dim

        dims   = [6+self.t_dim, *hidden, 6] # in= 6 coord+1 time
        layers = []

        for idx, (d_in, d_out) in enumerate(pairwise(dims)):
            layers                          += [nn.Linear(d_in, d_out)]
            if idx < len(dims) - 2: layers  += [nn.ReLU()] # -- add relu unless it's the last layer
        
        self.net    = nn.Sequential(*layers).to(device)
        self.opt    = optim.Adam(self.parameters(), lr=lr)
        self.device = device

    @property
    def name(self):
        axis_str    = 'a' if self.axis == 'all_axes' else self.axis
        colour_str  = 'a' if self.colour == 'all_colours' else self.colour
        return f"expert_{axis_str}{colour_str}"

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
        v_colour    = out[:, self.col_i]

        # -- create new tensor so shapes match up (output has out_dim shape, not 6)
        vel                 = torch.zeros_like(pts.reshape(-1, 6))
        vel[:, self.axis_i] = v_axis
        vel[:, self.col_i]  = v_colour

        return vel.view(B, N, 6)

    # ─────────────────────────────────────────  training  ──
    def train_loop(self,
                   loader: DataLoader,
                   n_epochs: int = 5,
                   ckpt_dir: str | Path | None = None):
        """
        Flow-matching objective for linear (rectified-flow) forward process:
            x_t = (1-t)·x₀ + t·ε        ⇒   v* = ε - x₀
        We sample ε once per batch; t ~ U(0,1).
        Loss is MSE on the **masked** dimensions only.
        """

        mask        = self.mask.to(self.device)            # [6]
        mse         = nn.MSELoss()

        for ep in tqdm(range(1, n_epochs + 1)):
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


# ───────────────────────────────────── demo run ──
if __name__ == "__main__":
    ds      = SphereDataset(num_datapoints=10_000)
    loader  = DataLoader(ds, batch_size=64, shuffle=True)   # whole cloud per batch
    CHECKPOINT_DIR = Path(__file__).parent / 'checkpoints'

    expert_xr  = Expert_FlowMatching(axis='x', colour='r', device='cuda:0')
    expert_yg  = Expert_FlowMatching(axis='y', colour='g', device='cuda:0')
    expert_zb  = Expert_FlowMatching(axis='z', colour='b', device='cuda:0')
    expert_all = Expert_FlowMatching(axis='all', colour='all', device='cuda:0')
    
    expert_xr .train_loop(loader, n_epochs=100, ckpt_dir=CHECKPOINT_DIR)
    expert_yg .train_loop(loader, n_epochs=100, ckpt_dir=CHECKPOINT_DIR)
    expert_zb .train_loop(loader, n_epochs=100, ckpt_dir=CHECKPOINT_DIR)
    expert_all.train_loop(loader, n_epochs=100, ckpt_dir=CHECKPOINT_DIR)

    expert_xr .load_state_dict(torch.load(CHECKPOINT_DIR / 'expert_xr.pt'))
    expert_yg .load_state_dict(torch.load(CHECKPOINT_DIR / 'expert_yg.pt'))
    expert_zb .load_state_dict(torch.load(CHECKPOINT_DIR / 'expert_zb.pt'))
    expert_all.load_state_dict(torch.load(CHECKPOINT_DIR / 'expert_all.pt'))

    noise = torch.randn(1000, 6, device='cuda:0')
