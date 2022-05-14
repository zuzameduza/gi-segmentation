import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import rle2mask


class GiMriDataset(Dataset):
    def __init__(self, imgs_df, csv_path, transform=None, load_labels=True):
        self.imgs_df = imgs_df
        self.gt_df = pd.read_csv(csv_path)
        self.transform = transform
        self.load_labels = load_labels

    def __len__(self):
        return len(self.imgs_df)

    def __getitem__(self, idx):
        img_path = self.imgs_df.iloc[idx]['path']
        img_id = self.imgs_df.iloc[idx]['id']

        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Read annotations
        if self.load_labels:
            # (mask with 4 classes (with background as 0))
            segmentation_classes = ['large_bowel', 'small_bowel', 'stomach']
            mask = np.zeros(img.shape)

            for i, seg_cls in enumerate(segmentation_classes):
                rle = self.gt_df.loc[(self.gt_df['id'] == img_id) &
                                     (self.gt_df['class'] == seg_cls)]['segmentation'].item()

                if type(rle) is str:
                    rle_m = rle2mask(rle, label=i + 1, shape=img.shape)
                    mask = np.where(rle_m >= 1, rle_m, mask)
        else:
            mask = None

        if self.transform:
            img = self.transform['img'](image=img)['image']
            mask = self.transform['mask'](image=mask)['image']

        mask = torch.squeeze(mask).type(torch.LongTensor)
        return (img, mask, img_id)

