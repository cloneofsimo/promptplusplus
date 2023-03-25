# Prompt++

<!-- #region -->
<p align="center">
<img  src="contents/diag.png">
</p>
<!-- #endregion -->

Unofficial Implementation of [Prompt+](https://prompt-plus.github.io/), with bit of my own additions to further explore the P+ space of the stable diffusion.

# Introduction

We typically use single text conditioning as an input. Naturally, we reuse the same conditioning for all CrossAttention layer. What if we don't? Prompt+ explores this idea, that we can use different text embedding for different cross attention layers, and it works! Prompt+ textual inversion allows you to "extend" the textual inversion process "per-layer", so although we get 16 total tokens, we can expect better results than the original textual inversion.

Ok, but the code was not released yet, so I decided to implement it myself. I also added some of my own ideas to further explore the P+ space of the stable diffusion.

## Usage

### Installation

As of now, this repo requires lora-diffusion as a dependency. You can install it by

```bash
pip install git+https://github.com/cloneofsimo/lora.git
```

(I will remove this dependency in the future, maybe...)

Install this repo by

```bash
pip install git+https://github.com/cloneofsimo/ppp.git
```

### Training

Use `ppp_train` command to train: following example

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./dataset/data_yc"
export OUTPUT_DIR="./exps/yc"

ppp_train --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --placeholder_tokens="<yc>" \
  --use_template="object" \
  --do_coarse_inversion=False \
  --preserve_prefix=False \
```

### Inference

Inference is bit tricky. You need to set attentionprocessor that I made, and overwrite the pipeline of `StableDiffusionPipeline` with custom call function. Luckily, I do that all for you. If you would like to know what is going on, please check out the source code.

```python

from ppp import PPPPromptManager
from ppp import overwrite_call
from ppp import PPPAttenProc
from lora_diffusion import patch_pipe

from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda"
)

pm = PPPPromptManager(tokenizer= pipe.tokenizer, \
    text_encoder=pipe.text_encoder, \
    main_token="<yc>", preserve_prefix=False, extend_amount=1)


pipe.unet.set_attn_processor(PPPAttenProc())


patch_pipe(pipe, "./exps/yc/step_inv_1000.safetensors")

with torch.no_grad():
    ps = pm.embed_prompt("a colorful photo of a <yc> in the jungles")
torch.manual_seed(0)
overwrite_call(pipe, prompt_embeds=ps).images[0].save("test.png")
```
