from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SphereDataset

class Expert_FlowMatching(nn.Module):

    AXIS = {'x': 0, 'y': 1, 'z': 2, 'r': 3, 'g': 4, 'b': 5}

    def __init__(self,
                 axis: str, colour: str,
                 hidden: list[int] = [128, 128],
                 lr: float = 1e-3, device: str = 'cpu'):
        super().__init__()
        assert axis in 'xyz' and colour in 'rgb'
        self.axis_i    = self.AXIS[axis]
        self.col_i     = self.AXIS[colour]
        self.mask      = torch.zeros(6, dtype=torch.bool)
        self.mask[self.axis_i] = self.mask[self.col_i] = True   # [6] bool

        dims  = [7, *hidden, 2]                    # in= 6 coord+1 time
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out)]
            if d_out != 2: layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers).to(device)
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.device = device

    @torch.no_grad()
    def calculate_velocity(self, pts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        pts : [B,N,6]  cloud batch
        t   : [B,1] or scalar
        Returns masked velocity [B,N,6] (zeros on unsupervised slots).
        """
        B, N, _ = pts.shape
        inp      = torch.cat([pts, t.expand(B, N, 1)], dim=-1)   # [B,N,7]
        v_ax, v_cl = self.net(inp.reshape(-1, 7)).split(1, dim=1)  # [B*N,1] each
        vel      = torch.zeros_like(pts.reshape(-1, 6))
        vel[:, self.axis_i]  = v_ax.squeeze(1)
        vel[:, self.col_i]   = v_cl.squeeze(1)
        return vel.view(B, N, 6)

    # ─────────────────────────────────────────  training  ──
    def train_loop(self,
                   loader: DataLoader,
                   n_epochs: int = 5,
                   ckpt_dir: str | Path | None = None):
        """
        Flow-matching objective for linear (rectified-flow) forward process:
            x_t = (1-t)·x₀ + t·ε        ⇒   v* = ε – x₀
        We sample ε once per batch; t ~ U(0,1).
        Loss is MSE on the **masked** dimensions only.
        """
        mask = self.mask.to(self.device)            # [6]
        mse  = nn.MSELoss()

        for ep in range(1, n_epochs + 1):
            tot, n = 0.0, 0
            for cloud in loader:                    # cloud [N,6]
                cloud   = cloud.to(self.device).unsqueeze(0)     # [1,N,6]
                B, N, _ = cloud.shape

                eps     = torch.rand_like(cloud)                 # ε
                t       = torch.rand(B, 1, 1, device=self.device)
                x_t     = (1 - t) * cloud + t * eps              # noisy

                target  = (eps - cloud)                          # v*
                pred    = self.calculate_velocity(x_t, t)

                diff    = (pred - target)[:, :, mask]            # [B,N,2]
                loss    = mse(diff, torch.zeros_like(diff))

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                tot += loss.item(); n += 1

            print(f"epoch {ep:02d}  loss {tot/n:.6f}")
            if ckpt_dir:
                Path(ckpt_dir).mkdir(exist_ok=True, parents=True)
                torch.save(self.state_dict(),
                           Path(ckpt_dir)/f"expert_{ep:02d}.pt")


# ───────────────────────────────────── demo run ──
if __name__ == "__main__":
    ds      = SphereDataset()
    loader  = DataLoader(ds, batch_size=1, shuffle=True)   # whole cloud per batch
    expert  = Expert_FlowMatching(axis='x', colour='r', device='cpu')
    expert.train_loop(loader, n_epochs=5)