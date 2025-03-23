from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from .era5 import ERA5
from .gridsat import GridSat



def load_data(
    *, channel, cond_channel,size_x,size_y,cond_x,cond_y, batch_size, image_size, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param deterministic: if True, yield results in a deterministic order.
    """
    dataset = ImageDataset(
        image_size,
        channel,
        cond_channel,
        size_x,
        size_y,
        cond_x,
        cond_y,
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
       
    while True:
        yield from loader


class ImageDataset(Dataset):
    def __init__(self, resolution, channel, cond_channel,size_x,size_y,cond_x,cond_y):
        super().__init__()
        self.resolution = resolution
        self.channel = channel
        self.cond_channel = cond_channel
        self.size_x = size_x
        self.size_y = size_y
        self.cond_x = cond_x
        self.cond_y = cond_y
        self.data_set = ERA5(split='train')
        self.condition_set = GridSat(split='train')

    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):
        data = self.data_set.__getitem__(idx)
        arr = np.stack(data, axis=0)
        arr = arr[:self.channel,:self.size_x,:self.size_y]
        for i in range (self.channel):
            min_val = np.min(arr[i,:,:])
            max_val = np.max(arr[i,:,:])
            arr[i,:,:] = (arr[i,:,:] - min_val) / (max_val - min_val)*2-1
        arr = arr.astype(np.float32)
     
        cond_data = self.condition_set.__getitem__(idx)
        con_arr = np.squeeze(cond_data)
        con_arr = np.nan_to_num(con_arr)
        con_arr = con_arr[:self.cond_channel,:self.cond_x,:self.cond_y]
        for i in range (self.cond_channel):
            min_val = np.min(con_arr[i,:,:])
            max_val = np.max(con_arr[i,:,:])
            con_arr[i,:,:] = (con_arr[i,:,:] - min_val) / (max_val - min_val)*2-1

        con_arr = con_arr.astype(np.float32)
        out_dict = {}
        return arr, con_arr, out_dict
    