from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler,DDIMScheduler
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from animatediff.models.unet import UNet3DConditionModel
from omegaconf import OmegaConf
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import load_weights
from safetensors import safe_open
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from faceadapter.face_adapter import FaceAdapterPlusForVideoLora
from diffusers.utils import load_image
from animatediff.utils.util import save_videos_grid
from huggingface_hub import hf_hub_download

def load_model():
    inference_config = "inference-v2.yaml"
    sd_version = "animatediff/sd"
    id_ckpt = "animator.ckpt"
    image_encoder_path = "image_encoder"
    dreambooth_model_path = "realisticVisionV60B1_v51VAE.safetensors"
    motion_module_path="mm_sd_v15_v2.ckpt"
    inference_config = OmegaConf.load(inference_config)    


    tokenizer    = CLIPTokenizer.from_pretrained(sd_version, subfolder="tokenizer",torch_dtype=torch.float16,
    )
    text_encoder = CLIPTextModel.from_pretrained(sd_version, subfolder="text_encoder",torch_dtype=torch.float16,
    ).cuda()
    vae          = AutoencoderKL.from_pretrained(sd_version, subfolder="vae",torch_dtype=torch.float16,
    ).cuda()
    unet = UNet3DConditionModel.from_pretrained_2d(sd_version, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)
    ).cuda()
    pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=None,
            #beta_start=0.00085, beta_end=0.012, beta_schedule="linear",steps_offset=1
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
            # scheduler=EulerAncestralDiscreteScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
            # scheduler=EulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="linear",steps_offset=1

    ),torch_dtype=torch.float16,
            ).to("cuda")
    
    pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path         = motion_module_path,
            motion_module_lora_configs = [],
            # domain adapter
            adapter_lora_path          = "",
            adapter_lora_scale         = 1,
            # image layers
            dreambooth_model_path      = None,
            lora_model_path            = "",
            lora_alpha                 = 0.8
    ).to("cuda")
    if dreambooth_model_path != "":
        print(f"load dreambooth model from {dreambooth_model_path}")
        dreambooth_state_dict = {}
        with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                dreambooth_state_dict[key] = f.get_tensor(key)
                        
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, pipeline.vae.config)
            # print(vae)
            #vae ->to_q,to_k,to_v
            # print(converted_vae_checkpoint)
            convert_vae_keys = list(converted_vae_checkpoint.keys())
            for key in convert_vae_keys:
                if "encoder.mid_block.attentions" in key or "decoder.mid_block.attentions" in  key:
                    new_key = None
                    if "key" in key:
                        new_key = key.replace("key","to_k")
                    elif "query" in key:
                        new_key = key.replace("query","to_q")
                    elif "value" in key:
                        new_key = key.replace("value","to_v")
                    elif "proj_attn" in key:
                        new_key = key.replace("proj_attn","to_out.0")
                    if new_key:
                        converted_vae_checkpoint[new_key] = converted_vae_checkpoint.pop(key)

            pipeline.vae.load_state_dict(converted_vae_checkpoint)

            converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, pipeline.unet.config)
            pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)

            pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict).to("cuda")
        del dreambooth_state_dict
        pipeline = pipeline.to(torch.float16)
        id_animator = FaceAdapterPlusForVideoLora(pipeline, image_encoder_path, id_ckpt, num_tokens=16,device=torch.device("cuda"),torch_type=torch.float16)
        return id_animator
if __name__ == "__main__":
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    from insightface.utils import face_align
    import cv2
    from PIL import Image
    animator = load_model()
    random_seed  = 4993
    prompt = "Iron Man soars through the clouds, his repulsors blazing"
    negative_prompt="semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    
    
    
    pil_image=[load_image("demos/lecun.png")]
    img_path = "demos/lecun.png"
    img = cv2.imread(img_path)
    faces = app.get(img)
    face_roi = face_align.norm_crop(img,faces[0]['kps'],112)
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    pil_image = [Image.fromarray(face_roi).resize((224, 224))]
    sample = animator.generate(pil_image, negative_prompt=negative_prompt,prompt=prompt,num_inference_steps = 30,seed=random_seed,
    guidance_scale      = 8,
                width               = 512,
                height              = 512,
                video_length        = 16,
                scale=0.8,
    )

    prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
    save_videos_grid(sample, f"outputdir/{prompt}-{random_seed}.gif")
