import os

import numpy as np
import pandas as pd


def get_matadata(data_path):
    metadata = {'id': [],
                'path': [],
                'case': [],
                'day': [],
                'slice': [],
                'img_height': [],
                'img_width': [],
                'pxl_height_mm': [],
                'pxl_width_mm': []}

    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:

            # Get image path
            img_path = os.path.join(dirname, filename)
            metadata['path'].append(img_path)

            # Sample path:
            # <dir>/case101/case101_day20/scans/slice_0001_266_266_1.50_1.50.png

            # Normalize and split the path
            img_path = os.path.normpath(img_path)
            splt_path = img_path.split(os.sep)

            # Get image case
            img_case = splt_path[-4][4:]
            metadata['case'].append(img_case)

            # Get image day
            img_day = splt_path[-3].split('_')[1][3:]
            metadata['day'].append(img_day)

            # slice, height, width, pxl spacings
            _, slice, img_height, img_width, pxl_height_mm, pxl_width_mm = splt_path[-1][:-4].split('_')
            metadata['slice'].append(slice)
            metadata['img_height'].append(img_height)
            metadata['img_width'].append(img_width)
            metadata['pxl_height_mm'].append(pxl_height_mm)
            metadata['pxl_width_mm'].append(pxl_width_mm)

            # Create proper id
            # Sample id: case123_day20_slice_0001
            img_id = splt_path[-3] + '_slice_' + slice
            metadata['id'].append(img_id)

    # Create and return df
    metadata_df = pd.DataFrame(metadata)
    return metadata_df


# from: https://ccshenyltw.medium.com/run-length-encode-and-decode-a33383142e6b
def rle2mask(mask_rle: str, label=1, shape=(266, 266)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)  # Needed to align to RLE direction


# from: https://ccshenyltw.medium.com/run-length-encode-and-decode-a33383142e6b
def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
