import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, Dataset
import numpy as np

from petrel_client.client import Client
import io
import xarray as xr
import pandas as pd
import os
import torch


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class EncoderNet(nn.Module):
    def __init__(self,channels, out_channels, use_conv=False, dims=2, multi = [32,64,32]):
        super(EncoderNet, self).__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.dims = dims
        self.multi = multi

        self.input_conv1 = conv_nd(self.dims, self.channels, self.multi[0], 3, padding=1)
        self.input_conv2 = conv_nd(self.dims, self.multi[0], self.multi[1], 3, padding=1)
        self.middle_conv = conv_nd(self.dims, self.multi[1], self.multi[1], 3, padding=1)
        self.output_conv1 = conv_nd(self.dims, self.multi[1], self.multi[2], 3, padding=1)
        self.output_conv2 = conv_nd(self.dims, self.multi[2], self.out_channels, 3, padding=1)
    
        if self.out_channels == self.channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        x_input = x
        x = self.input_conv1(x)
        x = self.input_conv2(x)
        x = self.middle_conv(x)
        x = self.output_conv1(x)
        x = self.output_conv2(x)
        return x

model = EncoderNet(3,3).to("cuda:0")
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

Years = {
    'train': ['2010-01-01 00:00:00', '2021-12-31 23:00:00'],
    'valid': ['2022-01-01 00:00:00', '2022-12-31 23:00:00'],
    'test': ['2023-01-01 00:00:00', '2023-06-30 23:00:00'],
    'all': ['2010-01-01 00:00:00', '2023-06-30 23:00:00']
}

vnames_list = ["irwin_cdr", "irwin_vza_adj", "irwvp"]

class GridSat(Dataset):
    def __init__(self, data_dir='', split='train', **kwargs) -> None:
        super().__init__()

        Years_dict = kwargs.get('years', Years)
        self.file_stride = kwargs.get('file_stride', 6)
        self.split = split
        self.data_dir = data_dir
        self.client = Client(conf_path="~/petreloss.conf")
        years = Years_dict[split]
        self.init_file_list(years)

    def init_file_list(self, years):
        time_sequence = pd.date_range(years[0],years[1],freq=str(self.file_stride)+'h')
        self.file_list = [os.path.join(str(time_stamp.year), f"GRIDSAT-B1.{time_stamp.strftime('%Y.%m.%d.%H')}.v02r01.nc") for time_stamp in time_sequence]

    def __len__(self):
        data_len = len(self.file_list)
        return data_len

    def __getitem__(self, index):
        file_path = os.path.join(self.data_dir, self.file_list[index])
        array_lst = []
        with io.BytesIO(self.client.get(file_path)) as f:
            nc_data = xr.open_dataset(f, engine='h5netcdf')
            for vname in vnames_list:
                D = nc_data.data_vars[vname].data
                array_lst.append(D[np.newaxis, :, :])
            data = np.concatenate(array_lst, axis=0)[:, :, :]
            array = data
        del array_lst
        return array

encoder_inputs = torch.randn(1, 3, 2000, 5143)

condition_set = GridSat(split='train')
lis = []
for i in range (100):
    cond_data = condition_set.__getitem__(i)
    con_arr = np.squeeze(cond_data)
    con_arr = np.nan_to_num(con_arr)
    for i in range (3):
        min_val = np.min(con_arr[i,:,:])
        max_val = np.max(con_arr[i,:,:])
        con_arr[i,:,:] = (con_arr[i,:,:] - min_val) / (max_val - min_val)*2-1
    con_arr = con_arr.astype(np.float32)

    con_arr = np.expand_dims(con_arr, axis=0)
    con_arr = torch.from_numpy(con_arr)
    con_arr = con_arr.to('cuda:0')
    lis.append(con_arr)

encoder_inputs = torch.cat(lis, dim=0)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = MyDataset(encoder_inputs)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        x_input = batch[0]
        x_input = torch.unsqueeze(x_input, dim=0).to("cuda:0")
        optimizer.zero_grad()
        
        output = model(x_input)
        
        loss = criterion(output, x_input)
        
        loss.backward()
        
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}")

save_path = ""
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")