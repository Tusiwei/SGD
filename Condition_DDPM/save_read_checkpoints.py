import torch
from petrel_client.client import Client
import os
import io

def save_checkpoint(model, config_setup, ckpt_dir, epoch, iter, record_metrics, optimizer, lr_scheduler, scaler, save_as_latest=True):
    save_states = {'model': model,
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'scaler': scaler.state_dict(),
                  'record_metrics': record_metrics,
                  'epoch': epoch,
                  'iter': iter,
                  'config': config_setup}

    save_path = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch}_{iter}.pth')
    client = Client(conf_path="/mnt/petrelfs/feiben/petreloss.conf")
    with io.BytesIO() as f:
        torch.save(save_states, f)
        client.put(save_path, f.getvalue())

if __name__ == "__main__":
    save_checkpoint()
