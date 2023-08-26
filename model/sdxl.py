from typing import Tuple

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer


def load_sdxl(
    checkpoint_path, revision=None
) -> Tuple[
    AutoTokenizer,
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoencoderKL,
    UNet2DConditionModel,
]:
    tokenizer1 = AutoTokenizer.from_pretrained(
        checkpoint_path,
        subfolder="tokenizer",
        use_fast=False,
        revision=revision,
    )
    tokenizer2 = AutoTokenizer.from_pretrained(
        checkpoint_path,
        subfolder="tokenizer_2",
        use_fast=False,
        revision=revision,
    )
    text_encoder1 = CLIPTextModel.from_pretrained(
        checkpoint_path,
        subfolder="text_encoder",
        revision=revision,
    )
    text_encoder2 = CLIPTextModelWithProjection.from_pretrained(
        checkpoint_path,
        subfolder="text_encoder_2",
        revision=revision,
    )
    vae = AutoencoderKL.from_pretrained(
        checkpoint_path,
        subfolder="vae",
        revision=revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        checkpoint_path,
        subfolder="unet",
        revision=revision,
    )
    return tokenizer1, tokenizer2, text_encoder1, text_encoder2, vae, unet
