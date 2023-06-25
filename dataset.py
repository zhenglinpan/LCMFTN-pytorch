from torch.utils.data import Dataset
import h5py
from glob import glob
import os
import cv2
import torchvision.transforms as transforms

class AnimeDataset(Dataset):
    def __init__(self, args, trans=None):
        super(AnimeDataset).__init__()
        self.trans = transforms.Compose(trans)
        
        pairs = []
        for root, dirs, files in './dataset':
            h5list = glob(os.path.join(root, dirs)+'/*.h5')
            if len(h5list > 0):
                for h5file in h5list:
                    with h5py.File(h5file, 'r') as hf:
                        for i in range(hf['n']):
                            pairs.append(hf[i])
        
        print(self.pairs[0]['Cn'].shape)
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