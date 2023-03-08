# ControlNet with Diffusers
## Requirements

~~~
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate safetensors
pip install xformers==0.0.17.dev466
~~~

### option

~~~
pip install controlnet_hinter
~~~

## How to use canny2image.py

~~~
python canny2image.py ^
  --model model\Counterfeit-V2.5 ^
  --vae vae\counterfeit_vae ^
  --H 1024 --W 1024 ^
  --prompt prompt.txt ^
  --threshold1 50 --threshold2 50 ^
  --image sample.jpg ^
  --n_samples 10
~~~

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

## How to inpaint.py

~~~
python inpaint.py ^
  --controlnet basemodel\sd-controlnet-openpose ^
  --image original_image.jpg ^
  --mask mask.png ^
  --hint pose.png ^
  --W 768 --H 768 ^
  --prompt prompt.txt ^
  --seed 40000 ^
  --n_samples 10
~~~

## For more details (link to my blog)

https://touch-sp.hatenablog.com/entry/2023/02/23/181611
