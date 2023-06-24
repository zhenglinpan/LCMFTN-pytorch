from torch.utils.data import Dataset

class AnimeDataset(Dataset):
    def __init__(self, args):
        super(AnimeDataset).__init__()