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
# import netCDF4
# from s3_client import s3_client
Years = {
    'train': ['2010-01-01 00:00:00', '2021-12-31 23:00:00'],
    'valid': ['2022-01-01 00:00:00', '2022-12-31 23:00:00'],
    'test': ['2023-01-01 00:00:00', '2023-06-30 23:00:00'],
    'all': ['2010-01-01 00:00:00', '2023-06-30 23:00:00']
}

'''
Years = {
    'train': ['2010-01-01 00:00:00', '2021-12-31 23:00:00'],
    'valid': ['2022-01-01 00:00:00', '2022-12-31 23:00:00'],
    'test': ['2023-01-01 00:00:00', '2023-06-30 23:00:00'],
    'all': ['2010-01-01 00:00:00', '2023-06-30 23:00:00']
}
'''

vnames_list = ["irwin_cdr", "irwin_vza_adj", "irwvp"]

class ERA5(Dataset):
    def __init__(self, data_dir='cluster2:s3://era5_np_float32/single', split='train', **kwargs) -> None:
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
        vari_lis = ['u10','v10','t2m','msl']

        self.file_list = [
            os.path.join(
                str(time_stamp.year),
                str(time_stamp.year) + '-' + str(time_stamp.month).zfill(2) + '-' + str(time_stamp.day).zfill(2),
                f"{time_stamp.strftime('%H')}:00:00-{vari}.npy"
            )
            for time_stamp in time_sequence
            for vari in vari_lis
        ]
        print(len(self.file_list))


    def __len__(self):
        data_len = int(len(self.file_list)/4)
        return data_len

    def __getitem__(self, index):
        arr_lst = []
        file_path = os.path.join(self.data_dir, self.file_list[index*4])
        with io.BytesIO(self.client.get(file_path)) as f:
            field_data = np.load(f)
            arr_lst.append(field_data)
            
        file_path = os.path.join(self.data_dir, self.file_list[index*4+1])
        with io.BytesIO(self.client.get(file_path)) as f:
            field_data = np.load(f)
            arr_lst.append(field_data)

        file_path = os.path.join(self.data_dir, self.file_list[index*4+2])
        with io.BytesIO(self.client.get(file_path)) as f:
            field_data = np.load(f)
            arr_lst.append(field_data)

        file_path = os.path.join(self.data_dir, self.file_list[index*4+3])
        with io.BytesIO(self.client.get(file_path)) as f:
            field_data = np.load(f)
            arr_lst.append(field_data)
        return arr_lst

if __name__ == "__main__":
    data_set = ERA5(split='train')
    for i in range (0,124):
        data = data_set.__getitem__(i)
        arr = np.stack(data, axis=0)
        print(arr.shape)
    print("complete")
