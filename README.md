# Face0 - SDXL
**(Work in progress)**

Unofficial implementation of [Face0](https://arxiv.org/abs/2306.06638) with [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)


![50000](https://github.com/bryandlee/face0-sdxl/assets/26464535/fa67fb98-d333-4d6e-9bac-9dfba49caf72)


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
* Use image-caption pairs instead of single fixed caption
* Train with LoRA or Adapter
* Apply EMA
* Implement guidance method described in section 2.2
* Add a legit inference script


## Credits
* Trainig code based on [diffusers](https://github.com/huggingface/diffusers)
* Inference pipeline from [kohya_ss](https://github.com/bmaltais/kohya_ss)
* Face encoder from [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
