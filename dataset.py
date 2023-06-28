import os
import cv2
import numpy as np
import h5py

from glob import glob

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from matplotlib import pyplot as plt

class AnimeDataset(Dataset):
    def __init__(self, args, trans=None):
        super(AnimeDataset).__init__()
        self.trans = transforms.Compose(trans)
        
        pairs = []
        h5_root = os.path.join(args.dataset_root, 'h5')
        h5_files = glob(h5_root + '/*.h5')
        for h5_f in h5_files:
            with h5py.File(h5_f, 'r') as hf:
                for i in range(len(list(hf.keys())) // 4):
                    pairs.append({'Sn': np.array(hf[f'{i}_Sn']).astype(np.uint8), 
                                  'Sp': np.array(hf[f'{i}_Sp']).astype(np.uint8), 
                                  'Cn': np.array(hf[f'{i}_Cn']).astype(np.uint8)[..., [2,1,0]], # RGB-BGR since opencv used
                                  'Cp': np.array(hf[f'{i}_Cp']).astype(np.uint8)[..., [2,1,0]]})
        
        print(f'{len(pairs)} pairs founded.')
        self.pairs = pairs
    
    def __getitem__(self, index):
        Cn = self.trans(self.pairs[index]['Cn'])
        Cp = self.trans(self.pairs[index]['Cp'])
        Sn = self.trans(self.pairs[index]['Sn'])
        Sp = self.trans(self.pairs[index]['Sp'])
        return {'Cn': Cn, 'Cp': Cp, 'Sn': Sn, 'Sp': Sp}
    
    def __len__(self):
        return len(self.pairs)