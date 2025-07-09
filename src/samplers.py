import  torch
from    torch import Tensor
from    math import sqrt
from    torch.types import Number

from    src.dataset import sample_noise
from    src.experts import DiscriminativeExpert_Color, BaseFlowExpert, AnalyticExpert_Monochrome


@torch.no_grad()
def euler_sampling(
    expert:             BaseFlowExpert | AnalyticExpert_Monochrome,
    n_points:           int   = 512,
    n_denoise_steps:    int   = 32,
    step_size:          float = 1.0,
    return_trajectory:  bool  = True,
) -> Tensor:
    # -- simple euler sampler for a single flow model
    expert.eval()

    x       = sample_noise(n_points)
    ts      = torch.linspace(1.0, 0.0, n_denoise_steps + 1)
    traj    = []

    for i in range(n_denoise_steps):
        t_hi = ts[i]        .view(1, 1, 1)
        t_lo = ts[i + 1]    .view(1, 1, 1)
        dt   = (t_lo - t_hi).item()

        # -- euler updates
        v_hi    = expert.calculate_velocity(x.unsqueeze(0), t_hi.item()).squeeze(0)
        x      += step_size * dt * v_hi
        traj   += [x.clone().clamp(0, 1)]

    return torch.stack(traj) if return_trajectory else traj[-1]


@torch.no_grad()
def annealed_importance_sampling(
    generative_experts:     list[BaseFlowExpert | AnalyticExpert_Monochrome],
    discriminative_experts: list[DiscriminativeExpert_Color],
    n_particles:        int   = 2,
    n_points:           int   = 512,
    n_denoise_steps:    int   = 128,    # number of denoising steps for each expert (i.e. number of intermediate denoised distributions)
    n_seek_steps:       int   = 3,      # number of mcmc (langevin or gibbs) steps per intermediate denoised distribution
    step_size:          float = 1.,     # coefficient to adjust magnitude of flow ODE updates 
    kappa:              float = 0.05,   # step size of MCMC transition kernel
    return_trajectory:  bool  = True,   # whether to return all intermediate denoising steps for the purposes of visualization
) -> Tensor:
    for e in generative_experts + discriminative_experts: e.eval()

    log_w   = torch.zeros(n_particles)
    ts      = torch.linspace(1.0, 0.0, n_denoise_steps + 1)
    x       = torch.stack([sample_noise(n_points) for _ in range(n_particles)], dim=0)
    traj    = []

    # -- denoise over intermediate distributions (lines 3->11)
    for i in range(n_denoise_steps):
        t_hi, t_lo  = ts[i].item(), ts[i + 1].item()
        dt          = (t_lo - t_hi) / n_seek_steps
        # -- mcmc initialization (line 5)
        weights     = (1.                               for _    in generative_experts) # this is in case each expert is for a particular region in the scene
        velocities  = (e.calculate_velocity(x, t_hi)    for e    in generative_experts)
        x          += step_size * dt * sum(v*w          for v, w in zip(velocities, weights))

        for _ in range(n_seek_steps): # -- mcmc refinement (lines 6->8)
            velocities  = (e.calculate_velocity(x, t_hi) for e    in generative_experts)
            scores      = (_score(x, v, t_hi) * w        for v, w in zip(velocities, weights))
            x          += (kappa**2) / 2 * sum(scores)
            x          +=  kappa         * torch.randn_like(x)

        # -- reward experts (line 10)
        log_w           += sum(d.score(x).to(log_w) for d in discriminative_experts)
        should_resample  = discriminative_experts and _effective_sample_size(log_w) < (n_particles / 2)
        
        if should_resample:     x, log_w = _multinomial_resample(x, log_w)
        if return_trajectory:   traj    += [x[log_w.argmax()].clone().clamp(0, 1)]

    if return_trajectory:   return torch.stack(traj)
    else:                   return x[log_w.argmax()].clamp(0, 1)


def _effective_sample_size(log_w: Tensor) -> Tensor:
    # -- looks at the values of each particle, and counts
    # only the ones that still contribute. we use this to
    # determine when we should resample from a distribution
    # defined over these particles.
    return (w := torch.exp(log_w)).sum().square() / (w.square().sum() + 1e-8)

def _multinomial_resample(particles: Tensor, log_w: Tensor) -> tuple[Tensor, Tensor]:
    # -- resample particles based on their log weights.
    # -- this is the "importance sampling" in annealed importance sampling (AIS)
    # from the paper
    probs   = torch.softmax(log_w, dim=0)
    idx     = torch.multinomial(probs, num_samples=len(probs), replacement=True)
    return particles[idx], 0. * log_w[idx] 

def _score(x: Tensor, v: Tensor, t: Number) -> Tensor:
    # -- in rectified flows, alpha = t and sigma = (1-t)
    # -- we can use this to convert between velocity and score
    # with simple linear transformations :)
    alpha, sigma        = t, 1 - t
    deterministic_drift = (1/alpha) * x
    flow_contribution   = -(sigma * (sigma - alpha) / alpha**2) * v
    return deterministic_drift + flow_contribution
