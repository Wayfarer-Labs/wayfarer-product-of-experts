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
N_POINTS             = 512
LOADER               = DataLoader(SphereDataset(n_pts=N_POINTS), batch_size=BATCH_SIZE, shuffle=True)


# -- load (or train) all experts
from experts import (GenerativeExpert_Axis, DiscriminativeExpert_Color,
GenerativeExpert_Color, DiscriminativeExpert_Ellipsoid,
GenerativeExpert_Monochrome)

expert_x   = GenerativeExpert_Axis(axis='x', device=DEVICE, n_epochs=N_EPOCHS)
expert_y   = GenerativeExpert_Axis(axis='y', device=DEVICE, n_epochs=N_EPOCHS)
expert_z   = GenerativeExpert_Axis(axis='z', device=DEVICE, n_epochs=N_EPOCHS)
expert_all = GenerativeExpert_Axis(axis='all_axes', device=DEVICE, n_epochs=N_EPOCHS)
# expert_rgb = GenerativeExpert_Color(color='rgb', device=DEVICE, n_epochs=25)
expert_mono = GenerativeExpert_Monochrome(device=DEVICE, n_epochs=200)

expert_red = DiscriminativeExpert_Color(color='r', device=DEVICE)
expert_grn = DiscriminativeExpert_Color(color='g', device=DEVICE)
expert_blu = DiscriminativeExpert_Color(color='b', device=DEVICE)
expert_whi = DiscriminativeExpert_Color(color='white', device=DEVICE)
expert_ell = DiscriminativeExpert_Ellipsoid(device=DEVICE)

for expert in tqdm((expert_x, expert_y, expert_z, expert_all, expert_mono), desc='Loading experts'):
    if (ckpt := CHECKPOINT_DIR / f"{expert.name}.pt").exists():
        expert.load_state_dict(torch.load(ckpt)) ; continue
    
    expert.train_loop(LOADER, ckpt_dir=CHECKPOINT_DIR)

# -- visualize experts individually (should not denoise the whole sphere properly)
from samplers import sample, sample_product_of_experts

# for expert in tqdm((expert_x, expert_y, expert_z, expert_all, expert_rgb), desc='Visualizing generative experts'):
for expert in tqdm((expert_mono,), desc='Visualizing generative experts'):
    # continue # testing 
    denoised_trajectory = sample(expert, n_denoise_steps=128, return_trajectory=True, device=DEVICE)
    vis.render_points_over_time(denoised_trajectory, path=VISUALIZATION_DIR / f"{expert.name}.gif", show_path=True)


# -- inference-time products-of-experts sampler
for expert_clr in tqdm((expert_ell, expert_red, expert_grn, expert_blu, expert_whi), desc='Visualizing discriminative experts'):
    denoised_trajectory: Tensor = sample_product_of_experts(
        generative_experts=[expert_x, expert_y, expert_z, expert_mono],
        discriminative_experts=[expert_clr],
        n_points=N_POINTS,
        n_denoise_steps=128,
        n_seek_steps=3,
        step_size=1.0,
        device=DEVICE,
        return_trajectory=True,
    )
    vis.render_points_over_time(denoised_trajectory, path=VISUALIZATION_DIR / f"{expert_clr.name}.gif", show_path=True)
