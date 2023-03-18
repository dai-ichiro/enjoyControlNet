# ControlNet with Diffusers
## Requirements

~~~
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate safetensors opencv-python
pip install xformers==0.0.17.dev466
~~~

### option 1

~~~
pip install controlnet-aux
~~~

### option 2

~~~
pip install rembg[gpu]
~~~

## How to use canny2image.py

~~~
python canny2image.py ^
  --model model\Counterfeit-V2.5 ^
  --vae vae\counterfeit_vae ^
  --prompt prompt.txt ^
  --image sample.jpg ^
~~~

#### For more details (link to my blog)

https://touch-sp.hatenablog.com/entry/2023/02/23/181611

## How to use multi_controlnet.py

~~~
python multi_controlnet.py ^
  --model model\anything-v4.0 ^
  --vae vae\any4_vae ^
  --net1 controlnet\sd-controlnet-canny ^
  --image1 canny_image.png ^
  --net2 controlnet\sd-controlnet-openpose ^
  --image2 pose_image.png
~~~

#### For more details (link to my blog)

https://touch-sp.hatenablog.com/entry/2023/03/13/141954

## How to use inpaint.py

~~~
python inpaint.py ^
  --controlnet controlnet\sd-controlnet-openpose ^
  --image original_image.jpg ^
  --mask mask.png ^
  --hint pose.png ^
  --W 768 --H 768 ^
  --prompt prompt.txt ^
  --seed 40000 ^
  --n_samples 10
~~~

#### For more details (link to my blog)

https://touch-sp.hatenablog.com/entry/2023/03/10/135439

