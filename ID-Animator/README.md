# Quick start
## Preparations
1. Setup repository
```
git clone https://github.com/ID-Animator/ID-Animator
cd ID-Animator
pip install -r requirements.txt
```
2. downloads checkpoints
- Download Stable Diffusion V1.5 and put them into **animatediff\sd**  https://huggingface.co/spaces/ID-Animator/ID-Animator/tree/main/animatediff/sd
- Download ID-Animator checkpoint https://huggingface.co/spaces/ID-Animator/ID-Animator/blob/main/animator.ckpt
- Download AnimateDiff checkpoint https://huggingface.co/spaces/ID-Animator/ID-Animator/blob/main/mm_sd_v15_v2.ckpt
- Download CLIP Image encoder https://huggingface.co/spaces/ID-Animator/ID-Animator/tree/main/image_encoder
- Download realisticVisionV60B1 https://huggingface.co/spaces/ID-Animator/ID-Animator/blob/main/realisticVisionV60B1_v51VAE.safetensors
## Inference scripts
run ```python infer.py```
## GradIO
run ```python app.py```
