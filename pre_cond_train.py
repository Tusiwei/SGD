"""
Train a diffusion model on images.
"""
import torch as th
import argparse

from Pre_Condition_DDPM import logger
from Pre_Condition_DDPM.image_datasets import load_data
from Pre_Condition_DDPM.resample import create_named_schedule_sampler
from Pre_Condition_DDPM.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from Pre_Condition_DDPM.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    device = th.device("cuda:0")
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        channel = int(args.channel),
        cond_channel = int(args.cond_channel),
        size_x = int(args.size_x),
        size_y = int(args.size_y),
        cond_x = int(args.cond_x),
        cond_y = int(args.cond_y),
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()
    logger.log("training complete")


def create_argparser():
    defaults = dict(
        channel = '',
        cond_channel = '',
        size_x = '',
        size_y = '',
        cond_x = '',
        cond_y = '',
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=200000,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=200,
        save_interval=5000,
        resume_checkpoint = '',
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
