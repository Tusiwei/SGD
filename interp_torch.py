import numpy as np
import pandas as pd
import torch
import json
import os
import torch.nn.functional as F
import xarray as xr
import tqdm

lon_range = [0., 360.]
lat_range = [90., -90]
input_resolution = 0.0625 * 4

def calculate_mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def calculate_mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def convert_longitude(lon):
    return (lon + 360) % 360


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


if __name__ == '__main__':
    for j in range (0,3):
        w5k_data = np.load('')[:,::6,:]
        w5k_data = torch.from_numpy(w5k_data).to("cuda:0")
        field_data = np.load('')

        if len(field_data.shape) == 3:
            field_data = field_data[np.newaxis, :, :, :]
        field_data = torch.from_numpy(field_data).to("cuda:0")
        field_data = field_data.permute(0, 2, 3, 1).to("cuda:0")
        interp_data = interp_field_to_stn(field_data, w5k_data[:, j, -2], w5k_data[:, j, -1])

        print('------------')
        print(' mae_u10 is %.8f' % ( calculate_mae(interp_data[:,0], w5k_data[:,j,0])))
        print(' mse_u10 is %.8f' % ( calculate_mse(interp_data[:,0], w5k_data[:,j,0])))

        print(' mae_v10 is %.8f' % ( calculate_mae(interp_data[:,1], w5k_data[:,j,1])))
        print(' mse_v10 is %.8f' % ( calculate_mse(interp_data[:,1], w5k_data[:,j,1])))

        print(' mae_t2m is %.8f' % ( calculate_mae(interp_data[:,2], w5k_data[:,j,2]+273.15)))
        print(' mse_t2m is %.8f' % ( calculate_mse(interp_data[:,2], w5k_data[:,j,2]+273.15)))

        print(' mae_msl is %.8f' % ( calculate_mae(interp_data[:,3]/100, w5k_data[:,j,3])))
        print(' mse_msl is %.8f' % ( calculate_mse(interp_data[:,3]/100, w5k_data[:,j,3])))
