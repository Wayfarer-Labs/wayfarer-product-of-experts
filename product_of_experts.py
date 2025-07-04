from    tqdm import tqdm
from    dataset import SphereDataset
import  visualize as vis

import  torch
from    torch import Tensor
from    pathlib import Path
from    torch.utils.data import DataLoader


N_EPOCHS, BATCH_SIZE = 500, 64
DEVICE               = 'cuda:0'
CHECKPOINT_DIR       = Path(__file__).parent / 'checkpoints'
VISUALIZATION_DIR    = Path(__file__).parent / 'visualizations'
LOADER               = DataLoader(SphereDataset(n_pts=128), batch_size=BATCH_SIZE, shuffle=True)


# -- load (or train) all experts
from expert import Expert_FlowMatching

expert_xr  = Expert_FlowMatching(axis='x', colour='r', device=DEVICE, n_epochs=N_EPOCHS)
expert_yg  = Expert_FlowMatching(axis='y', colour='g', device=DEVICE, n_epochs=N_EPOCHS)
expert_zb  = Expert_FlowMatching(axis='z', colour='b', device=DEVICE, n_epochs=N_EPOCHS)
expert_all = Expert_FlowMatching(axis='all_axes', colour='all_colours', device=DEVICE, n_epochs=N_EPOCHS)

for expert in tqdm((expert_xr, expert_yg, expert_zb, expert_all), desc='Loading experts'):
    if (ckpt := CHECKPOINT_DIR / f"{expert.name}.pt").exists():
        expert.load_state_dict(torch.load(ckpt)) ; continue
    
    expert.train_loop(LOADER, ckpt_dir=CHECKPOINT_DIR)

# -- visualize experts individually (should not denoise the whole sphere properly)
from samplers import sample

groundtruth_sphere  = next(iter(LOADER))[0].to(DEVICE)
noise               = torch.randn_like(groundtruth_sphere, device=DEVICE)

for expert in tqdm((expert_xr, expert_yg, expert_zb, expert_all), desc='Visualizing experts'):

    denoised_trajectory: Tensor = sample(expert, x_init=noise, return_trajectory=True, device=DEVICE)
    vis.render_points_over_time(denoised_trajectory, VISUALIZATION_DIR / f"{expert.name}.gif")

exit()


# -- inference-time products-of-experts sampler
exit()
NUM_DENOISING_STEPS = 32
NUM_CLOUD_POINTS    = 512

intermediate_clouds = torch.empty(NUM_DENOISING_STEPS, *vis.sample_random_cloud(NUM_CLOUD_POINTS).shape)

def product_of_experts(cloud: Tensor, *experts: Expert_FlowMatching) -> Tensor:
    
    return torch.tensor([])

# -- render

OUTPUT_RENDER_PATH = Path(__file__).parent / 'expert_trajectories.gif'

vis.render_points_over_time(intermediate_clouds, OUTPUT_RENDER_PATH)


