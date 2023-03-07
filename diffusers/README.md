# ControlNet with Diffusers
## Requirements

~~~
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate safetensors
pip install controlnet_hinter
pip install xformers==0.0.17.dev466
~~~

## For more details (link to my blog)

https://touch-sp.hatenablog.com/entry/2023/02/23/181611

## How to use multi_controlnet.py

~~~
python multi_controlnet.py ^
  --model model\anything-v4.0 ^
  --vae vae\any4_vae ^
  --controlnet1 basemodel\sd-controlnet-canny ^
  --controlnet1_image canny_image.png ^
  --controlnet2 basemodel\sd-controlnet-openpose ^
  --controlnet2_image pose_image.png
~~~
