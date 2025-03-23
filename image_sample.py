"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch
import torch as th
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import sys

from Condition_DDPM import dist_util, logger
from Condition_DDPM.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Condition_DDPM.era5_2001x4000 import era5_2001x4000
from Condition_DDPM.era5 import ERA5
from Condition_DDPM.gridsat import GridSat

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

class GaussianBlur(nn.Module):
    def __init__(self, kernel):
        super(GaussianBlur, self).__init__()
        self.kernel_size = len(kernel)
        assert self.kernel_size % 2 == 1, 'kernel size must be odd.'
        self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=self.kernel, requires_grad=False)
 
    def forward(self, x):
        x1 = x[:,0,:,:].unsqueeze_(1)
        x2 = x[:,1,:,:].unsqueeze_(1)
        x3 = x[:,2,:,:].unsqueeze_(1)
        padding = self.kernel_size // 2
        x1 = F.conv2d(x1, self.weight)
        x2 = F.conv2d(x2, self.weight)
        x3 = F.conv2d(x3, self.weight)
        x = torch.cat([x1, x2, x3], dim=1)
        return x
    

def get_gaussian_blur(kernel_size, device):
    kernel = gkern(kernel_size, 2).astype(np.float32)
    gaussian_blur = GaussianBlur(kernel)
    return gaussian_blur.to(device)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cuda:0")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []

    def blur_cond_fn(x, t,weight=None,corner=None,scale=None, y=None, x_lr=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            loss = 0
            if not x_lr is None:
                device_x_in_lr = x_in.device
                x_lr = x_lr[:, :, int(1/scale*corner[0]):int(1/scale*(corner[0]+corner[2])), int(1/scale*corner[1]):int(1/scale*(corner[1]+corner[3]))]
                x_in_lr=x_in
                
                weight = nn.Parameter(weight)
                weight.requires_grad_()
                in_channels = 4
                conv_layer = nn.Conv2d(in_channels, in_channels, kernel_size=9, padding=4,stride=scale, bias=False,groups=4)
                conv_layer.weight = nn.Parameter(weight)
                
                x_in_lr = conv_layer((x_in_lr+1)/2)

                x_lr = x_lr.to(device_x_in_lr) 
                x_lr = (x_lr + 1) / 2
                mse = (x_in_lr - x_lr) ** 2
                mse = mse.mean(dim=(1,2,3))
                mse = mse.sum()

                loss = -mse * 5000
                
                conv_layer.zero_grad()
                mse.backward(retain_graph=True)
                grad_conv_kernel = conv_layer.weight.grad
                weight = weight- grad_conv_kernel * 0.05
                
                print('step t %d, mse is %.8f' % (t[0], mse*5000))
            return weight, th.autograd.grad(loss, x_in)[0]
    def model_fn(x, cond, t, y=None):
        if y==None:
            y=1
        return model(x, cond, t, y if args.class_cond else None)

    data_set_2 = GridSat(split='test')
    data_set_1 = ERA5(split='test')
    for j in range (0,128):     
        cond = data_set_2.__getitem__(j)
        cond = np.squeeze(cond)
        cond = np.nan_to_num(cond)
        cond = cond[:,:2000,:5143]
        for i in range (3):
            min_val = np.min(cond[i,:,:])
            max_val = np.max(cond[i,:,:])
            cond[i,:,:] = (cond[i,:,:] - min_val) / (max_val - min_val)*2-1
        cond = cond[np.newaxis, :, :, :]
        cond = torch.from_numpy(cond)
        cond = cond.to("cuda:0")

        era5_data = data_set_1.__getitem__(j)
        lr = np.stack(era5_data, axis=0)
        lr = lr[:,:720,:]

        lr_c = np.copy(lr)
        lr_c = np.expand_dims(lr_c, axis=0)

        for i in range (4):
            min_val = np.min(lr[i,:,:])
            max_val = np.max(lr[i,:,:])
            lr[i,:,:] = (lr[i,:,:] - min_val) / (max_val - min_val)*2-1
        lr = np.expand_dims(lr, axis=0)
        lr = torch.from_numpy(lr)
        weight = torch.zeros(4, 1, 9, 9)
        weight = weight.to("cuda:0")
        weight = weight.float()

        scale = int(args.scale)

        cond_fn = lambda x,t,weight,corner,scale,y=1 : blur_cond_fn(x, t,weight=weight,corner=corner,scale=scale, y=y, x_lr=lr)
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            weight,
            cond,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            scale=scale,
        )
        sample = sample.clamp(-1,1)
        sample = sample.cpu().numpy()
        
        for i in range (4):
            min_sample = np.min(sample[:,i,:,:])
            max_sample = np.max(sample[:,i,:,:])
            sample[:,i,:,:] = (sample[:,i,:,:] - min_sample) / (max_sample - min_sample)
        

        for i in range (4):
            min_val = np.min(lr_c[:,i,:,:])
            max_val = np.max(lr_c[:,i,:,:])
            sample[:,i,:,:] = ((sample[:,i,:,:]+1)/2) * (max_val - min_val) + min_val
        np.save(str(j)+'.npy',sample)
        print("sample is complete.")

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        scale='1',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
