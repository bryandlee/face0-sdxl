import os
import shutil
import logging
import itertools
from typing import Dict

import fire
import torch
import transformers

import diffusers
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from tqdm import tqdm

from data.dataset import build_dataset, get_data_iter, get_collate
from model import build_encoders, load_sdxl
from common.util import get_function_args


logger = get_logger(__name__)


def build_optimizer(_target_, params, **kwargs):
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "adadelta": torch.optim.Adadelta,
        "adagrad": torch.optim.Adagrad,
        "adamax": torch.optim.Adamax,
        "asgd": torch.optim.ASGD,
        "rmsprop": torch.optim.RMSprop,
        "rprop": torch.optim.Rprop,
        "adafactor": transformers.optimization.Adafactor,
    }
    optimizer_cls = optimizers[_target_]
    return optimizer_cls(params, **kwargs)


def compute_snr(timesteps, noise_scheduler):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def train(
    base_model_path: str,
    dataset: Dict,
    image_encoder: Dict,
    image_size: int = 768,
    logdir: str = "../runs",
    train_steps: int = 100_000,
    batch_size: int = 1,
    lr_unet: float = 1e-4,
    lr_projection: float = 1e-4,
    lr_image_backbone: float = 1e-4,
    train_image_backbone: bool = False,
    dataloader_num_workers: int = 0,
    gradient_accumulation_steps: int = 1,
    mixed_precision: str = "bf16",
    seed: int | None = 14,
    allow_tf32: bool = True,
    optimizer: Dict | None = None,
    scheduler: Dict | None = None,
    max_grad_norm: float = 1.0,
    checkpoint_interval: int | str = 1000,
    checkpoints_total_limit: int | None = 10,
    log_interval: int = 20,
    continue_checkpoint: str | None = None,
    snr_gamma: float | None = None,
):
    args = get_function_args()

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    checkpoint_interval_path = checkpoint_interval
    if isinstance(checkpoint_interval_path, str):
        with open(checkpoint_interval_path, "r") as f:
            checkpoint_interval = int(f.read())

    # Set accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=logdir,
        logging_dir=logdir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    os.makedirs(logdir, exist_ok=True)
    if accelerator.is_main_process:
        OmegaConf.save(args, os.path.join(logdir, "config.yaml"))
        accelerator.init_trackers("trackers")

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(logdir, "train.log"),
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if seed is not None:
        set_seed(seed)

    # Build dataset
    dataset = build_dataset(**dataset)
    logger.info(f"Dataset Info:\n{dataset}")

    # Load model
    tokenizer1, tokenizer2, text_encoder1, text_encoder2, vae, unet = load_sdxl(base_model_path)

    tokenizers = [tokenizer1, tokenizer2]
    text_encoders = [text_encoder1, text_encoder2]

    vae.requires_grad_(False)
    text_encoder1.requires_grad_(False)
    text_encoder2.requires_grad_(False)
    unet.requires_grad_(True)

    vae.eval()
    text_encoder1.eval()
    text_encoder2.eval()
    unet.train()

    vae_dtype = torch.float32
    text_encoder_dtype = torch.bfloat16
    unet_dtype = torch.float32

    vae.to(accelerator.device, dtype=vae_dtype)
    text_encoder1.to(accelerator.device, dtype=text_encoder_dtype)
    text_encoder2.to(accelerator.device, dtype=text_encoder_dtype)
    unet.to(accelerator.device, dtype=unet_dtype)

    unet.enable_xformers_memory_efficient_attention()

    noise_scheduler = DDPMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

    image_backbone, image_projection = build_encoders(**image_encoder)

    if train_image_backbone:
        image_backbone.train()
        image_backbone.to(dtype=torch.float32, device=accelerator.device)
        for param in image_backbone.parameters():
            param.requires_grad_(True)
    else:
        image_backbone.eval()
        image_backbone.to(dtype=torch.float32, device=accelerator.device)
        for param in image_backbone.parameters():
            param.requires_grad_(False)

    image_projection.train()
    image_projection.to(dtype=torch.float32, device=accelerator.device)
    for param in image_projection.parameters():
        param.requires_grad_(True)

    # Build optimizer, scheduler
    params_to_optimize = [
        {
            "params": unet.parameters(),
            "lr": lr_unet,
        },
        {
            "params": image_projection.parameters(),
            "lr": lr_projection,
        },
    ]
    if train_image_backbone:
        params_to_optimize.append(
            {
                "params": image_backbone.parameters(),
                "lr": lr_image_backbone,
            }
        )

    optimizer = build_optimizer(params=params_to_optimize, **optimizer or {})

    lr_scheduler = get_scheduler(
        optimizer=optimizer,
        **scheduler,
    )

    if continue_checkpoint:
        unet_checkpoint = os.path.join(continue_checkpoint, "pytorch_bin")
        unet.load_state_dict(torch.load(unet_checkpoint, map_location="cpu"))

        face_projection_checkpoint = os.path.join(continue_checkpoint, "pytorch_model_1.bin")
        image_projection.load_state_dict(torch.load(face_projection_checkpoint, map_location="cpu"))

        optimizer_checkpoint = os.path.join(continue_checkpoint, "optimizer.bin")
        optimizer_state_dict = optimizer.state_dict()
        for k, v in torch.load(optimizer_checkpoint, map_location="cpu")["state"].items():
            optimizer_state_dict["state"][k] = v
        optimizer.load_state_dict(optimizer_state_dict)

        face_encoder_checkpoint = os.path.join(continue_checkpoint, "pytorch_model_2.bin")
        if os.path.exists(face_encoder_checkpoint):
            image_backbone.load_state_dict(torch.load(face_encoder_checkpoint, map_location="cpu"))

        logger.info(f"Loaded checkpoint from {continue_checkpoint}")

    # Build dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=get_collate(tokenizer=tokenizers),
        num_workers=dataloader_num_workers,
    )

    # Prepare with accelerator
    if train_image_backbone:
        (
            unet,
            image_projection,
            image_backbone,
            optimizer,
            dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet, image_projection, image_backbone, optimizer, dataloader, lr_scheduler
        )
    else:
        unet, image_projection, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, image_projection, optimizer, dataloader, lr_scheduler
        )

    data_iter = get_data_iter(dataloader)

    # Train
    progress_bar = tqdm(range(train_steps), disable=not accelerator.is_local_main_process, desc="Steps")
    step = 0
    while True:
        batch = next(data_iter)

        with accelerator.accumulate(image_projection):
            image = batch["image"].to(dtype=vae.dtype, device=vae.device)
            bsize = len(image)

            face = batch["face"].to(dtype=torch.float32, device=accelerator.device)
            face_embeds = image_projection(image_backbone(face))

            # Convert images to latent space
            model_input = vae.encode(image).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor
            model_input = model_input.to(unet_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=model_input.device,
            )
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            # Tokens to text embeds
            prompt_embeds_list = []
            for token_ids_i, text_encoder_i in zip(batch["token_ids"], text_encoders):
                prompt_embeds_out = text_encoder_i(
                    token_ids_i.to(text_encoder_i.device),
                    output_hidden_states=True,
                )
                pooled_prompt_embeds = prompt_embeds_out[0]
                prompt_embeds = prompt_embeds_out.hidden_states[-2]
                prompt_embeds = prompt_embeds.view(bsize, prompt_embeds.shape[1], -1)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bsize, -1)

            # XXX: does not reflect the real size and crop coords for now
            original_size = (image.shape[2], image.shape[3])
            target_size = (image.shape[2], image.shape[3])
            crops_coords_top_left = (0, 0)
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])
            add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype).repeat(
                bsize, 1
            )

            # Drop face embeddings with prob 0.1
            override = torch.rand(bsize) > 0.1
            prompt_embeds[override, -face_embeds.shape[1] :] = face_embeds[override].to(
                dtype=prompt_embeds.dtype
            )

            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={
                    "time_ids": add_time_ids,
                    "text_embeds": pooled_prompt_embeds,
                },
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")

            if "mask" in batch and batch["mask"] is not None:
                if step == 0:
                    logger.info("Using mask")
                masks = batch["mask"].to(accelerator.device)
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(1).float(),
                    size=loss.shape[-2:],
                    mode="nearest",
                )
                loss = loss * masks

            if snr_gamma is not None:
                snr = compute_snr(timesteps)
                mse_loss_weights = (
                    torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                )
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights

            loss = loss.mean()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                if train_image_backbone:
                    params_to_clip = itertools.chain(
                        unet.parameters(), image_projection.parameters(), image_backbone.parameters()
                    )
                else:
                    params_to_clip = itertools.chain(unet.parameters(), image_projection.parameters())
                accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            step += 1

            if accelerator.is_main_process:
                if step % checkpoint_interval == 0:
                    checkpoint_dir = os.path.join(logdir, "ckpt")
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if checkpoints_total_limit is not None:
                        checkpoints = os.listdir(checkpoint_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(checkpoint_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(checkpoint_dir, f"checkpoint-{step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if step % 10 == 0:
                    try:
                        if isinstance(checkpoint_interval_path, str):
                            with open(checkpoint_interval_path, "r") as f:
                                checkpoint_interval = int(f.read())
                        logger.info(f"checkpoint interval updated: {checkpoint_interval}")
                    except:
                        logger.info(
                            f"Could not read checkpoint interval from {checkpoint_interval_path}"
                        )

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        if step % log_interval == 0:
            accelerator.log(logs, step=step)

        if step >= train_steps:
            break

    accelerator.end_training()


def train_from_yaml(config_path: str):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    train(**config)


if __name__ == "__main__":
    fire.Fire()
