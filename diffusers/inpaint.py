import torch
from stable_diffusion_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline

from diffusers import ControlNetModel, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained("basemodel/sd-controlnet-openpose", torch_dtype=torch.float16)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "model/stable-diffusion-inpainting", 
    controlnet=controlnet, 
    safety_checker=None, 
    torch_dtype=torch.float16)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

image = load_image('out.png').resize((512, 512))
mask_image = load_image('pose_mask.png').resize((512, 512))
controlnet_conditioning_image = load_image('pose_result.png').resize((512, 512))

image = pipe(
    "a beautiful woman wearing high neck sweater, best quality, extremely detailed",
    image,
    mask_image,
    controlnet_conditioning_image,
    negative_prompt='monochrome, lowres, bad anatomy, worst quality, low quality',
    num_inference_steps=50,
    width=512,
    height=512
).images[0]

image.save("out.png")