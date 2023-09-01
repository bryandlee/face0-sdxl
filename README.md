# Face0 - SDXL

Unofficial implementation of [Face0](https://arxiv.org/abs/2306.06638) with [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)


![grid](https://github.com/bryandlee/face0-sdxl/assets/26464535/2baac317-959e-40fc-96de-7bede63d8980)


## Train
```shell
accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    train.py train_from_yaml \
    --config_path=configs/train.yaml
```

## Inference
* See `notebooks/inference.ipynb`


## TODO
* Train with LoRA or Adapter
* Apply EMA
* Implement guidance method described in section 2.2
* Add a legit inference script


## Credits
* Training code based on [diffusers🧨](https://github.com/huggingface/diffusers)
* Inference pipeline from [kohya_ss](https://github.com/bmaltais/kohya_ss)
* Face encoder from [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
