import  torch
from    experts import GenerativeExpert_Axis, DiscriminativeExpert_Color

@torch.no_grad()
def sample(
    expert: GenerativeExpert_Axis,
    n_points: int             = 512,
    n_denoise_steps: int      = 32,
    step_size: float          = 1.0,
    device:       str         = 'cpu',
    return_trajectory: bool = False,
) -> torch.Tensor:
    # -- simple euler sampler for a single flow model
    expert.eval()

    x       = torch.rand(n_points, 6, device=device)
    ts      = torch.linspace(1.0, 0.0, n_denoise_steps + 1, device=device)
    traj    = []

    for i in range(n_denoise_steps):
        t_hi = ts[i].view(1, 1, 1)
        t_lo = ts[i + 1].view(1, 1, 1)
        dt   = (t_lo - t_hi).item()

        # -- euler updates
        v_hi = expert.calculate_velocity(x.unsqueeze(0), t_hi).squeeze(0)
        x = x + (step_size * dt * v_hi)

        traj += [x.clone().clamp(0, 1)]

    return torch.stack(traj) if return_trajectory else traj[-1]

def _effective_sample_size(log_w):
    # -- looks at the values of each particle, and counts
    # only the ones that still contribute. we use this to
    # determine when we should resample from a distribution
    # defined over these particles.
    w = torch.exp(log_w)
    return w.sum().square() / (w.square().sum() + 1e-8)


def _multinomial_resample(particles, log_w):
    # -- resample particles based on their log weights
    # -- this is the "importance sampling" in annealed importance sampling (AIS)
    # from the paper
    probs   = torch.softmax(log_w, dim=0)
    idx     = torch.multinomial(probs, num_samples=len(probs), replacement=True)
    return particles[idx], 0. * log_w[idx] 


def sample_product_of_experts(
    generative_experts:     list[GenerativeExpert_Axis],
    discriminative_experts: list[DiscriminativeExpert_Color],
    n_particles: int          = 6,
    n_points: int             = 512,
    n_denoise_steps: int      = 32,     # number of denoising steps for each expert (i.e. number of intermediate denoised distributions)
    n_seek_steps: int         = 3,      # number of mcmc steps per intermediate denoised distribution
    step_size: float          = 1.0,    # coeff to adjust ode updates 
    device:       str         = 'cpu',
    return_trajectory: bool = False, # whether to return all intermediate denoising steps for the purposes of visualization
) -> torch.Tensor:
    for e in generative_experts + discriminative_experts: e.eval()

    x       = torch.rand(n_particles, n_points, 6, device=device)
    log_w   = torch.zeros(n_particles, device=device)
    ts      = torch.linspace(1.0, 0.0, n_denoise_steps + 1, device=device)

    trajectory = [x[log_w.argmax()].clone().clamp(0, 1)] if return_trajectory else []

    # -- denoise over intermediate distributions
    for i in range(n_denoise_steps):
        t_hi = ts[i]       .view(1, 1, 1).expand(n_particles,-1,-1)
        t_lo = ts[i + 1]   .view(1, 1, 1).expand(n_particles,-1,-1)

        dt   = (t_lo - t_hi) / n_seek_steps

        for _ in range(n_seek_steps): # -- mcmc refinement
            # -- sum velocities from all generative experts
            v = sum(g.calculate_velocity(x, t_hi) for g in generative_experts)
            # -- euler update
            x = x + (step_size * dt * v)

        # -- reward experts
        log_w           += sum(d.score(x).to(log_w) for d in discriminative_experts)
        should_resample  = discriminative_experts and _effective_sample_size(log_w) < n_particles / 2
        
        if should_resample:     x, log_w = _multinomial_resample(x, log_w)
        if return_trajectory:   trajectory.append(x[log_w.argmax()].clone().clamp(0, 1))

    if return_trajectory:   return torch.stack(trajectory)
    else:                   return x[log_w.argmax()].clamp(0, 1)
