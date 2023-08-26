# Face0 - SDXL
Unofficial implementation of [Face0](https://arxiv.org/abs/2306.06638) with [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (Work in progress)


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
