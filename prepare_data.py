import torch
import os, shutil
import h5py
import cv2
from args import args
from PIL import Image
from huggingface_hub import HfApi
from pathlib import Path
import numpy as np
from controlnet_aux import LineartDetector
from matplotlib import pyplot as plt
from tqdm import tqdm

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

def prepare_h5(args):
    '''
    https://huggingface.co/lllyasviel/control_v11p_sd15_lineart
    CAUTION WITH RAM CAPACITY !!!
    '''
    processor = LineartDetector.from_pretrained("./models/lineart_anime")
    
    save_dir = os.path.join(args.dataset_root, 'h5')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    color_previous = []
    color_current = []
    sketch_previous = []
    sketch_current = []
    h5_file_counter = 0
    save_gap = 1000
    for root, dirs, files in os.walk(args.shots_root):
        print(f"{len(files)} files found.")
        for i, file in tqdm(enumerate(files), total=len(files)):
            if file.endswith('jpg'):
                concate_image = cv2.imread(os.path.join(root, file))    # cv2: array: H W C, adjust resolution for model
                left_img = concate_image[:, :concate_image.shape[1]//2, :]
                right_img = concate_image[:, concate_image.shape[1]//2:, :]
                
                left_img = cv2.resize(left_img, (256, 192)) # W, H
                right_img = cv2.resize(right_img, (256, 192))
                
                left_sketch = processor(left_img, detect_resolution=192, image_resolution=192, return_pil=False)
                right_skecth = processor(right_img, detect_resolution=192, image_resolution=192, return_pil=False)
                
                color_previous.append(left_img)
                color_current.append(right_img)
                sketch_previous.append(left_sketch[..., 0][..., None])   # 3to1 channel
                sketch_current.append(right_skecth[..., 0][..., None])
            
            if (i+1) % save_gap == 0 or i == len(files) - 1:
                h5_file_counter += 1
                with h5py.File(os.path.join(save_dir, f'pairs{h5_file_counter}_{i+1}.h5'), 'w') as hf:
                    for j in range(len(color_previous)):
                        hf.create_dataset(f'{j}_Cp', data=color_previous[j])
                        hf.create_dataset(f'{j}_Cn', data=color_current[j])
                        hf.create_dataset(f'{j}_Sp', data=sketch_previous[j])
                        hf.create_dataset(f'{j}_Sn', data=sketch_current[j])
                    color_previous = []
                    color_current = []
                    sketch_previous = []
                    sketch_current = []
                print(f"\nh5 file has been saved to {os.path.join(save_dir, f'pairs_{h5_file_counter}_{len(color_previous)}.h5')}.")
    print("All Done.")

if __name__ == '__main__':
    prepare_h5(args)