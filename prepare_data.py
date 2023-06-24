import torch
import os
import h5py
import cv2
from args import args
from PIL import Image

def prepare_data(args, stride, width):
    """Extract frames from a video and save them as JPG files."""
    if not os.path.exists('./dataset/shot0'):
        os.makedirs('./dataset/shot0')

    video = cv2.VideoCapture(args.video_dir)
    
    frame_pairs = []
    frame_count = 0
    frame_pre_count = 0
    while True:
        ret, frame = video.read()
        frame_count += 1
        if not ret: 
            break
        
        pairs = {'Sn':None, 'Sp': None, 'Cn': None, 'Cp': None}
        if frame_count % stride == 0:
            pairs['Cp'] = frame
            frame_pre_count = frame_count
        if frame_count - frame_pre_count == width:
            pairs['Cn'] = frame
            frame_pre_count = frame_count
        if pairs['Cp'] is not None and pairs['Cn'] is not None:
            frame_pairs.append(pairs)
    video.release()
    
    for pair in frame_pairs:
        pair['Sn'] = model(pair['Cn']).cpu().detach().numpy()[0]
        pair['Sp'] = model(pair['Cp']).cpu().detach().numpy()[0]
        
    # TODO: add model and set monitor
    
    i = 0
    with h5py.File('./dataset/shot0/train.h5', 'w') as hf:
        hf.create_dataset(f'{i}', data=frame_pairs[i])
    

if __name__ == '__main__':
    prepare_data(args)