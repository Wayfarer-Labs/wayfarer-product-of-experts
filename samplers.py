import math
import torch
from typing import Optional

from expert import Expert_FlowMatching

@torch.no_grad()
def sample(
    expert: Expert_FlowMatching,
    n_points: int             = 512,
    n_steps: int              = 32,
    step_size: float          = 1.0,
    device:       str         = 'cpu',
    x_init: Optional[torch.Tensor] = None,
    return_trajectory: bool = False,
) -> torch.Tensor:
    """
    Deterministic Euler (or Heun) integration of the learned rectified-flow
    velocity field from t=1 → t=0.

    Parameters
    ----------
    n_points     : how many points in the cloud
    n_steps      : number of time steps (larger = better, slower)
    step_size    : global multiplier on the velocity (ε in paper, 0.5-1.5)
    x_init       : optional custom noise tensor [N,6]; if None ~ U(0,1)
    second_order : if True uses Heun (predictor-corrector) instead of Euler

    Returns
    -------
    cloud : [N,6] float tensor in [0,1]
    """
    expert.eval()                                   # inference mode

    x  = (torch.rand(n_points, 6, device=device)    # start from noise
            if x_init is None else x_init.to(device))

    # linear time grid   t_0 = 1 … t_n = 0
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

    traj = []

    for i in range(n_steps):
        t_hi = ts[i].view(1, 1, 1)                 # shape [1,1,1]
        t_lo = ts[i + 1].view(1, 1, 1)
        dt   = (t_lo - t_hi).item()                # negative

        # ---- predictor (Euler) ----
        v_hi = expert.calculate_velocity(x.unsqueeze(0), t_hi).squeeze(0)
        x = x + (step_size * dt * v_hi)

        traj += [x.clone().clamp(0, 1)]

    return torch.stack(traj) if return_trajectory else traj[-1]
