cd ..
mkdir ckpt
cd ckpt

mkdir stable-diffusion-xl-base-1.0
cd stable-diffusion-xl-base-1.0
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/model_index.json

mkdir scheduler
cd scheduler
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/scheduler/scheduler_config.json
cd ..

mkdir text_encoder
cd text_encoder
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/config.json
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/model.safetensors
cd ..

mkdir text_encoder_2
cd text_encoder_2
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/config.json
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.safetensors
cd ..

mkdir tokenizer
cd tokenizer
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer/merges.txt
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer/special_tokens_map.json
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer/tokenizer_config.json
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer/vocab.json
cd ..

mkdir tokenizer_2
cd tokenizer_2
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer_2/merges.txt
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer_2/special_tokens_map.json
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer_2/tokenizer_config.json
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer_2/vocab.json
cd ..

mkdir unet
cd unet
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/config.json
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/diffusion_pytorch_model.safetensors
cd ..

mkdir vae
cd vae
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae/config.json
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae/diffusion_pytorch_model.safetensors
cd ..

