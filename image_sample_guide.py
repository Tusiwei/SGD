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

import pandas as pd
import json
import xarray as xr
import tqdm




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
from Condition_DDPM.era5 import ERA5
from Condition_DDPM.gridsat import GridSat

lon_range = [0., 360.]
lat_range = [90., -90]
input_resolution = 0.0625 * 4

def calculate_mse_np(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def calculate_mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def calculate_mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def calculate_mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def convert_longitude(lon):
    return (lon + 360) % 360

def interp_field_to_stn_np(data, lon, lat):
    _, H, W, _ = data.shape
    
    if lon.min() < 0:
        lon = convert_longitude(lon)
    in_lon = lon_range[0] + np.array(range(W)) * input_resolution
    in_lat = lat_range[0] - np.array(range(H)) * input_resolution
    result = []
    for i in range(data.shape[-1]):
        tmp_data = xr.DataArray(data=data[0,:,:,i], dims=['y','x'], coords=(in_lat.tolist(), in_lon.tolist()))
        interp_data = tmp_data.interp(x=xr.DataArray(lon, dims='z'),y=xr.DataArray(lat, dims='z'))
        result.append(interp_data.data)
    return result


def interp_field_to_stn(data, lon, lat):
    _, H, W, _ = data.shape
    
    if lon.min() < 0:
        lon = convert_longitude(lon)
    
    in_lon = lon_range[0] + torch.arange(W, device=data.device) * input_resolution
    in_lat = lat_range[0] - torch.arange(H, device=data.device) * input_resolution
    
    Z = lon.shape[0]
    result = torch.zeros((Z, data.shape[-1]), device=data.device, dtype=data.dtype)
    
    for i in range(data.shape[-1]):
        tmp_data = data[0, :, :, i]
        grid_x, grid_y = torch.meshgrid(in_lon, in_lat, indexing='xy')
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).unsqueeze(0)
        grid[:, :, 0] = 2.0 * (grid[:, :, 0] - lon.min()) / (lon.max() - lon.min()) - 1.0
        grid[:, :, 1] = 2.0 * (grid[:, :, 1] - lat.min()) / (lat.max() - lat.min()) - 1.0
        
        lon_grid = 2.0 * (lon - lon.min()) / (lon.max() - lon.min()) - 1.0
        lat_grid = 2.0 * (lat - lat.min()) / (lat.max() - lat.min()) - 1.0
        sample_grid = torch.stack([lon_grid, lat_grid], dim=-1).unsqueeze(0).unsqueeze(0)
        
        interp_data = torch.nn.functional.grid_sample(tmp_data.unsqueeze(0).unsqueeze(0), sample_grid, align_corners=True, mode='bilinear')
        
        result[:, i] = interp_data.squeeze()
    
    return result

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
                # x_lr and x_in are of shape BChw, BCHW, they are float type that range from -1 to 1, x_in for small t'

                device_x_in_lr = x_in.device
                x_lr = x_lr[:, :, int(1/scale*corner[0]):int(1/scale*(corner[0]+corner[2])), int(1/scale*corner[1]):int(1/scale*(corner[1]+corner[3]))]
                x_in_lr=x_in
                
                weight = nn.Parameter(weight)
                weight.requires_grad_()
                in_channels = 4
                conv_layer = nn.Conv2d(in_channels, in_channels, kernel_size=9, padding=4,stride=scale, bias=False,groups=4)
                conv_layer.weight = nn.Parameter(weight)
                
                x_in_lr = conv_layer((x_in_lr+1)/2)
      
                field_data_np = x_in.detach().cpu().numpy()
                field_data_np = np.clip(field_data_np, -1, 1)
                field_cond_np = x_lr.detach().cpu().numpy()
                for i in range(4):
                    min_val = np.min(field_cond_np[:, i, :, :])
                    max_val = np.max(field_cond_np[:, i, :, :])
                    field_data_np[:, i, :, :] = ((field_data_np[:, i, :, :] + 1) / 2) * (max_val - min_val) + min_val
                field_data_np = np.transpose(field_data_np, (0,2,3,1))

                for i in range (4):
                    min_val = np.min(field_cond_np[:,:,:,i])
                    max_val = np.max(field_cond_np[:,:,:,i])
                    min_field = np.min(field_data_np[:,:,:,i])
                    max_field = np.max(field_data_np[:,:,:,i])

                    field_data_np[:,:,:,i] = ((field_data_np[:,:,:,i] - min_field) / (max_field - min_field)) * (max_val - min_val) + min_val
                w5k_data_np = np.load('')[:,::6,:]
                interp_data_np = interp_field_to_stn_np(field_data_np, w5k_data_np[:, j, -2], w5k_data_np[:, j, -1])
                print('step t %d, mse_u10_np is %.8f, mae_u10_np is %.8f' % (t[0], calculate_mse_np(interp_data_np[0], w5k_data_np[:,j,0]), calculate_mae_np(interp_data_np[0], w5k_data_np[:,j,0])))
                

                field_data = x_in                
                field_data = field_data.clamp(-1,1)
                for i in range(4):
                    min_val = torch.min(x_lr[:, i, :, :])
                    max_val = torch.max(x_lr[:, i, :, :])
                    field_data[:, i, :, :] = ((field_data[:, i, :, :] + 1) / 2) * (max_val - min_val) + min_val
                for i in range (4):
                    min_val = torch.min(x_lr[:,i,:,:])
                    max_val = torch.max(x_lr[:,i,:,:])
                    min_field = torch.min(field_data[:,i,:,:])
                    max_field = torch.max(field_data[:,i,:,:])

                    field_data[:,i,:,:] = ((field_data[:,i,:,:] - min_field) / (max_field - min_field)) * (max_val - min_val) + min_val
                field_data = field_data.permute(0, 2, 3, 1).to("cuda:0")
                w5k_data = np.load('')[:,::6,:]
                w5k_data = torch.from_numpy(w5k_data).to("cuda:0")
                interp_data = interp_field_to_stn(field_data, w5k_data[:, j, -2], w5k_data[:, j, -1])
                print('step t %d, mse_u10 is %.8f, mae_u10 is %.8f' % (t[0], calculate_mse(interp_data[:,0], w5k_data[:,j,0]), calculate_mae(interp_data[:,0], w5k_data[:,j,0])))

                x_lr = x_lr.to(device_x_in_lr) 
                x_lr = (x_lr + 1) / 2
                mse = (x_in_lr - x_lr) ** 2
                mse = mse.mean(dim=(1,2,3))
                mse = mse.sum()

                loss = -mse *6000
                
                conv_layer.zero_grad()
                mse.backward(retain_graph=True)
                grad_conv_kernel = conv_layer.weight.grad
                weight = weight- grad_conv_kernel * 0.05     
                print('step t %d, mse is %.8f' % (t[0], mse*6000))
            return weight, th.autograd.grad(loss, x_in)[0]
        
    def model_fn(x, cond, t, y=None):
        if y==None:
            y=1
        return model(x, cond, t, y if args.class_cond else None)

    data_set_2 = GridSat(split='test')
    data_set_1 = ERA5(split='test')
    for j in range (len(data_set_2)):     
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

        weight = torch.tensor(gkern(9,2))
        weight = torch.unsqueeze(weight, dim=0)
        weight = weight.repeat(4, 1, 1, 1)
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
