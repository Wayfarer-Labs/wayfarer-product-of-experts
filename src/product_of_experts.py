# %% (1) imports and constants
import  torch
from    tqdm.auto import tqdm
from    pathlib import Path
from    torch.utils.data import DataLoader

import  visualize as vis
from    ipynb_utils import HorizontalVisBuffer
from    dataset import SphereDataset


N_EPOCHS, BATCH_SIZE  = 500, 64
N_POINTS              = 512
N_DENOISE_STEPS       = 128
N_SEEK_STEPS          = 3
N_PARTICLES           = 8
STEP_SIZE             = 1.0
 
LOADER                = DataLoader(SphereDataset(n_pts=N_POINTS), batch_size=BATCH_SIZE, shuffle=True)
CHECKPOINT_DIR        = Path().cwd().parent / 'checkpoints'
VISUALIZATION_DIR     = Path().cwd().parent / 'visualizations'
DEVICE                = 'cuda:0' if torch.cuda.is_available() else 'cpu'
horizontal_vis_buffer = HorizontalVisBuffer(frames_per_gif=N_DENOISE_STEPS, fps=12)

torch.set_default_device(DEVICE)

# %% (2) instantiate all experts
# -- load (or train) all experts
from experts import GenerativeExpert_Axis, AnalyticExpert_Monochrome

expert_x    = GenerativeExpert_Axis(axis='x', n_epochs=N_EPOCHS)
expert_y    = GenerativeExpert_Axis(axis='y', n_epochs=N_EPOCHS)
expert_z    = GenerativeExpert_Axis(axis='z', n_epochs=N_EPOCHS)
expert_mono = AnalyticExpert_Monochrome()

# -- these models provide sa reward for 'particles' (aka clouds) of a certain color
from experts import DiscriminativeExpert_Color

discriminator_red   = DiscriminativeExpert_Color(color='r')
discriminator_green = DiscriminativeExpert_Color(color='g')
discriminator_blue  = DiscriminativeExpert_Color(color='b')

# %% (3) load or train all experts
for expert in tqdm((expert_x, expert_y, expert_z, expert_mono), desc='Loading experts'):
    if (ckpt := CHECKPOINT_DIR / f"{expert.name}.pt").exists():
        expert.load_state_dict(torch.load(ckpt)) ; continue
    
    expert.train_loop(LOADER, ckpt_dir=CHECKPOINT_DIR)

# %% (4) visualize experts individually

# -- these should not denoise the whole sphere in time
from samplers import euler_sampling, annealed_importance_sampling

generative_experts = [expert_x, expert_y, expert_z, expert_mono]
vis_progress_bar   = tqdm(range(100 * len(generative_experts)), desc='Visualizing generative experts...')

for expert in generative_experts:
    denoised_trajectory = euler_sampling(expert=expert)
    frames              = vis.render_points_over_time(denoised_trajectory, show_path=True, progress_bar=vis_progress_bar)
    horizontal_vis_buffer.push(expert.name, frames)

horizontal_vis_buffer.display() ; horizontal_vis_buffer.clear()

# %% (5) visualize generative experts in combination with a discriminator that rewards configurations with a certain color
# -- inference-time products-of-experts sampler
discriminative_experts    = [discriminator_red, discriminator_green, discriminator_blue]
vis_progress_bar          = tqdm(range(100 * len(discriminative_experts)), desc='Visualizing discriminative experts...')

for expert_color in discriminative_experts:
    denoised_trajectory = annealed_importance_sampling(generative_experts, [expert_color])
    frames              = vis.render_points_over_time(denoised_trajectory, show_path=True, progress_bar=vis_progress_bar)
    horizontal_vis_buffer.push(expert_color.name, frames)

horizontal_vis_buffer.display() ; horizontal_vis_buffer.clear()