'''https://huggingface.co/lllyasviel/control_v11p_sd15_lineart'''

import torch
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from controlnet_aux import LineartDetector

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

checkpoint = "ControlNet-1-1-preview/control_v11p_sd15_lineart"

image = load_image(
    "https://huggingface.co/ControlNet-1-1-preview/control_v11p_sd15_lineart/resolve/main/images/input.png"
)
image = image.resize((512, 512))

# prompt = "michael jackson concert"
processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

control_image = processor(image)
control_image.save("./images/control.png")

# controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
# )

# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()

# generator = torch.manual_seed(0)
# image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

# image.save('images/image_out.png')
