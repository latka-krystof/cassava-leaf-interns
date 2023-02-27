import torch
import pandas as pd
from PIL import Image as img
import torchvision.transforms as t
import constants

class StartingDataset(torch.utils.data.Dataset):
    """
    Load 800 x 600 images
    """

    def __init__(self, csv_path, im_path, training_set=True):

        df = pd.read_csv(csv_path)
        self.image_id = df['image_id']
        self.labels = df['label']
        self.im_path = im_path

        if training_set:
            self.image_id = self.image_id[:constants.TRAIN_NUM]
            self.labels = self.labels[:constants.TRAIN_NUM]
        else:
            self.image_id = self.image_id[constants.TRAIN_NUM: ]
            self.labels = self.labels[constants.TRAIN_NUM: ]

    def __getitem__(self, index):

        id = self.image_id.iloc[index]
        label = torch.tensor(int(self.labels.iloc[index]))

        img_path = constants.IMG_PATH + id

        image = img.open(img_path).resize((800, 600))
        
        return (t.ToTensor()(image), label)

    def __len__(self):

        return len(self.labels)

