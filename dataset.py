import os
import cv2
import numpy as np
import h5py

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class AnimeDataset(Dataset):
    def __init__(self, args, trans=None):
        super(AnimeDataset).__init__()
        self.trans = transforms.Compose(trans)
        
        pairs = []
        folders = os.listdir('./dataset')
        for folder in folders:
            if folder.startswith('shot'):
                h5file = os.path.join('./dataset', folder, 'pairs.h5')
                with h5py.File(h5file, 'r') as hf:
                    for i in range(len(list(hf.keys())) // 4):
                        pairs.append({'Sn': np.array(hf[f'{i}_Sn']).astype(np.uint8), 
                                      'Sp': np.array(hf[f'{i}_Sp']).astype(np.uint8), 
                                      'Cn': np.array(hf[f'{i}_Cn']).astype(np.uint8)[..., [2,1,0]],     # transpose since opencv used
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