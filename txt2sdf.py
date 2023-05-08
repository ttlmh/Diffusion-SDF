import argparse
import os

import torch
from omegaconf import OmegaConf
from pytorch3d.io import IO
from tqdm import trange

from models.voxdiff.models.diffusion.ddim import DDIMSampler
from models.voxdiff.util import instantiate_from_config
from utils.qual_util import save_mesh_as_gif
from utils.util_3d import init_mesh_renderer, sdf_to_mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out_path",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2sdf-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--save_obj",
        action='store_true',
        help="if saving the mesh files",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=6,
        help="how many samples to produce for the given prompt",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        help="the prompt to render"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/voxdiff-uinu.yaml",
        help="config path"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="ckpt/voxdiff-uinu.ckpt",
        help="model path"
    )

    opt = parser.parse_args()

    config = OmegaConf.load(opt.config_path)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(opt.model_path)["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    dist, elev, azim = 1.7, 20, 20
    mesh_renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=device)

    sampler = DDIMSampler(model)

    os.makedirs(opt.out_path, exist_ok=True)
    prompt = opt.prompt

    sample_path = os.path.join(opt.out_path)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples = list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])
            for n in trange(opt.n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(opt.n_samples * [prompt])
                shape = [8, 8, 8, 8]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)

    tar_dir = os.path.join(sample_path, f'{prompt.replace(" ", "-")}')
    os.makedirs(tar_dir, exist_ok=True)
    gen_mesh = sdf_to_mesh(x_samples_ddim)
    gen_gif_name = os.path.join(tar_dir, f'{os.path.basename(opt.model_path)}.gif')
    save_mesh_as_gif(mesh_renderer, gen_mesh, nrow=3, out_name=gen_gif_name)
    if opt.save_obj:
        for k, mesh in enumerate(gen_mesh):
            mesh_name = os.path.join(tar_dir,
                                     f'{prompt.replace(" ", "-")}_{os.path.basename(opt.model_path)}_{str(k)}.obj')
            IO().save_mesh(mesh, mesh_name)
