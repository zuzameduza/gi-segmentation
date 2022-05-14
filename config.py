import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch


DATA_DIR = '/home/zuza/kaggle/data/gi-segm/uw-madison-gi-tract-image-segmentation'
WORKDIR = '/home/zuza/kaggle/workdir/gi-segm'

BATCH_SIZE = 8
NUM_WORKERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
TRAINING_TRANSFORM = {'img': A.Compose([A.ToFloat(max_value=65535.0),
                                        A.Resize(156, 156),
                                        ToTensorV2()]),
                      'mask': A.Compose([A.Resize(156, 156, interpolation=cv2.INTER_NEAREST),
                                         ToTensorV2()])}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 25