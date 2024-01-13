import json

import utility.transcoding

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


args = utility.transcoding.args


class APIDataSet(Dataset):
    def __init__(self):
        super(APIDataSet, self).__init__()
        file_path: str = args.training_data_path
        number: int = 0
        with open(file=file_path, mode='r') as fp:
            for _ in tqdm(fp, desc='load dataset'):
                number += 1
        fp = open(file=file_path, mode='r')
        lines = fp.readlines()
        self.size: int = number
        self.file = lines

    def __len__(self):
        return self.size

    def __getitem__(self, item_idx):
        line = self.file[item_idx]

        data = json.loads(line.strip('\n'))
        label = data['label']
        data = utility.transcoding.encode_data(data)

        return data, label

