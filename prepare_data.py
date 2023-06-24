import os
import h5py
from args import args
from PIL import Image

def prepare_data(args):
    """
    read frame pairs and convert to h5
    """
    dataset_root = ''
    image_folder = ''

    s_imgs_namelist = [] # image name list
    c_img_namelist = [] # image name list
    for image_name in os.listdir(image_folder):
        if image_name.strip('-')[0] == '1':
            s_imgs_namelist.append(image_name)
        elif image_name.strip('-')[0] == '3':
            c_img_namelist.append(image_name)
    
    # TODO: sort c_imgs and s_imgs in numerical order, image name example: 1-0000.jpg
    s_imgs = []
    c_imgs = []
    # TODO: use PIL to read images from s_imgs_namelist and c_img_namelist then save to s_imgs and c_imgs
    # TODO: use h5py to save s_imgs and c_imgs to 2 .h5 files.


def extract_frames(args, model, stride, width):
    """
    read video and save frame pairs as jpg
    """
    # model = LeNet(pretrained=True)
    pass

if __name__ == '__main__':
    extract_frames(args, stride=5, width=40)
    prepare_data(args)