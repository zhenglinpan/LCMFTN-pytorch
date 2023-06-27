import torch
import os
import h5py
import cv2
from args import args
from PIL import Image
from huggingface_hub import HfApi
from pathlib import Path
import numpy as np
from controlnet_aux import LineartDetector
from matplotlib import pyplot as plt

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

# def prepare_data(args, stride, width, model=None):
#     """Extract frames from a video and save them as JPG files."""
#     if not os.path.exists('./dataset/shot0'):
#         os.makedirs('./dataset/shot0')

#     video = cv2.VideoCapture(args.video_dir)
    
#     frame_pairs = []
#     frame_count = 0
#     frame_pre_count = 0
#     while True:
#         ret, frame = video.read()
#         frame_count += 1
#         if not ret: 
#             break
        
#         pairs = {'Sn':None, 'Sp': None, 'Cn': None, 'Cp': None}
#         if frame_count % stride == 0:
#             pairs['Cp'] = frame
#             frame_pre_count = frame_count
#         if frame_count - frame_pre_count == width:
#             pairs['Cn'] = frame
#             frame_pre_count = frame_count
#         if pairs['Cp'] is not None and pairs['Cn'] is not None:
#             frame_pairs.append(pairs)
#     video.release()
    
#     for pair in frame_pairs:
#         pair['Sn'] = model(pair['Cn']).cpu().detach().numpy()[0]
#         pair['Sp'] = model(pair['Cp']).cpu().detach().numpy()[0]
#     # TODO: add model and set monitor
    
#     with h5py.File(os.path.join('./dataset/shot0', 'pairs.h5'), 'w') as hf:
#         for i, pair in enumerate(frame_pairs):
#             hf.create_dataset(f'{i}', data=pair)

def prepare_data(args, stride=0, width=0):
    def extract_frames(args):
        videolist = os.listdir(args.video_root)
        for videoname in videolist:
            if not os.path.exists(os.path.join('./dataset', videoname.split('.')[0])):
                os.makedirs(os.path.join('./dataset', videoname.split('.')[0]))
            video = cv2.VideoCapture(os.path.join(args.video_root, videoname))

            frame_count = 0
            while True:
                ret, frame = video.read()
                if ret is False: 
                    print(f'video {os.path.join(args.video_root, videoname)} break')
                    break
                frame = frame[:, 246:1672]  # cropping
                frame_name = os.path.join('./dataset', videoname.split('.')[0]) + '/' + '3-' + str(frame_count).zfill(4) + '.jpg'
                cv2.imwrite(frame_name, frame)
                frame_count += 1
            video.release()
    
    def convert():
        '''
        https://huggingface.co/lllyasviel/control_v11p_sd15_lineart
        '''
        processor = LineartDetector.from_pretrained("./models/lineart_anime")
        
        for root, dirs, files in os.walk('./dataset'):
            if len(files) > 0 and files[0].endswith('jpg'):
                for file in files:
                    if file.startswith('3-'):
                        image = cv2.imread(os.path.join(root, file))
                        # image = cv2.resize(image, (200, 150))   # cv2: array: W H C, adjust resolution for model
                        
                        control_image = processor(image)
                        control_image.save(os.path.join(root, file).replace('3-', '1-'))

    def pairing(args, stride=1, width=2):
        for root, dirs, files in os.walk('./dataset'):
            if len(files) > 0 and files[0].endswith('jpg'):
                if 'pairs.h5' in files: 
                    files.remove('pairs.h5')
                
                C_images = []
                S_images = []

                files = sorted(files, key=lambda f: int(f.strip('.jpg').split('-')[-1]))
                for file in files:
                    if file.endswith('jpg'):
                        if file.startswith('3-'):
                            C_images.append(cv2.imread(os.path.join(root, file)))
                        elif file.startswith('1-'):
                            S_images.append(cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE))
                assert len(C_images) == len(S_images)
                print(files)
                
                frame_pairs = []
                assert width <= len(C_images)
                for frame_count in range(width - 1, len(C_images), stride):
                    pair = {'Sn':None, 'Sp': None, 'Cn': None, 'Cp': None}
                    pair['Cp'] = C_images[frame_count - width + 1]
                    pair['Sp'] = S_images[frame_count - width + 1][:, :, None]
                    pair['Cn'] = C_images[frame_count]
                    pair['Sn'] = S_images[frame_count][:, :, None]
                    frame_pairs.append(pair)
                    
                print(len(frame_pairs))
                with h5py.File(os.path.join(root, 'pairs.h5'), 'w') as hf:
                    for i, pair in enumerate(frame_pairs):
                        hf.create_dataset(f'{i}_Cp', data=pair['Cp'])
                        hf.create_dataset(f'{i}_Sp', data=pair['Sp'])
                        hf.create_dataset(f'{i}_Cn', data=pair['Cn'])
                        hf.create_dataset(f'{i}_Sn', data=pair['Sn'])
                    
            
    # print("Extracting frames======================>")
    # extract_frames(args)
    # print("Converting======================>")
    # convert()
    print("Paring======================>")
    pairing(args)
    print("All Done!======================>")
    

if __name__ == '__main__':
    prepare_data(args)