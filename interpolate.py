# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import cv2
import dnnlib as dnnlib
import numpy as np
import PIL.Image
import torch

import legacy as legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--steps', type=int, help='Number of frames between images', default=10, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--out-name', 'output_name', help='Name of the video created', type=str, required=True)
def interpolate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    steps: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    output_name: str
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # creating the random vectors to perform the interpolation on
    zs = []
    for seed in seeds:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        zs.append(z)

    # creating the interpolation array
    out = []
    for i in range(len(zs)-1):
      for index in range(steps):
        fraction = index/float(steps)
        if fraction == 0:
            for _ in range(5):
                out.append(zs[i+1]*fraction + zs[i]*(1-fraction))
        else:
            out.append(zs[i+1]*fraction + zs[i]*(1-fraction))

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    # Generate images.
    img_list = []
    for z_idx, z in enumerate(out, start=1):
        print('Generating image for seed (%d/%d) ...' % (z_idx, len(out)))
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode, force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{z_idx:04d}.png')
        pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        temp_img = pil_img.copy()
        img_list.append(temp_img)

    # creatin the images for the video
    reversed_img_list = img_list.copy()
    reversed_img_list.reverse()
    img_list_for_video = [*img_list, *reversed_img_list]

    # Creating the video
    print("Creating the video output.")
    videodims = (512,512)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = 30
    video = cv2.VideoWriter(f'{outdir}/{output_name}.mp4',fourcc, fps,videodims)
    for img in img_list_for_video:
        video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    video.release()
      

#----------------------------------------------------------------------------

if __name__ == "__main__":

    interpolate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
