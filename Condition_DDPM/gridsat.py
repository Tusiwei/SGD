from torch.utils.data import Dataset
from petrel_client.client import Client
from tqdm import tqdm
import numpy as np
import io
import time
import xarray as xr
import json
import pandas as pd
import os
import gc
from multiprocessing import Pool
from multiprocessing import shared_memory
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import copy
import queue
import torch



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
        print(len(self.file_list))

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


if __name__ == "__main__":
    a = 0
    b = 124
    data_set = GridSat(split='valid')
    for i in range (a,b):
        try:
            data = data_set.__getitem__(i)
            print(data.shape)
            data = np.squeeze(data)
            data = np.nan_to_num(data)
            print(data.shape)
            path = os.path.join(str(i)+'.npy')
            print(np.sum(data, axis=(1, 2)))
            np.save(path, data)
        except:
            print(i)
    print("complete")
