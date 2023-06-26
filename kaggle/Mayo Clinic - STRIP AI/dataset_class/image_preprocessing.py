import seam_carving
import gc
import skimage, rasterio
import pandas as pd
import numpy as np
import albumentations as A

from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from rasterio.enums import Resampling
from albumentations.pytorch import ToTensorV2


def kfold(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """ KFold """
    fold = KFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    df['fold'] = -1
    for num, (tx, vx) in enumerate(fold.split(df)):
        df.loc[vx, "fold"] = int(num)
    return df


def mls_kfold(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """ Multilabel Stratified KFold """
    tmp_df = df.copy()
    y = pd.get_dummies(data=tmp_df.iloc[:, 2:8], columns=tmp_df.columns[2:8])
    fold = MultilabelStratifiedKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    for num, (tx, vx) in enumerate(fold.split(X=df, y=y)):
        df.loc[vx, "fold"] = int(num)
    del tmp_df
    gc.collect()
    return df


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data_folder from csv file like as train.csv, test.csv, val.csv
    """
    df = pd.read_csv(data_path)
    return df


def load_image(img_path: str, img_scaler: int = 1024) -> np.ndarray:
    """
    Load image with prefix image scaler
    Args:
        img_path: path to image
        img_scaler: image scaler default 1024 (1k)
    """
    image = rasterio.open(img_path)
    image = image.read(
        out_shape=(image.count, int(img_scaler), int(img_scaler)),
        resampling=Resampling.bilinear
    ).transpose(1, 2, 0)
    return image


def remove_background(image: np.ndarray) -> np.ndarray:
    """
    Apply Seam-Carving Function which is remove background from load image
    Args:
        image: np.ndarray from load_image function
    """
    image = seam_carving.resize(
        image, (image[1] - 512, image[0] - 512),
        energy_mode='backward',
        order=('width-first'),
        keep_mask=None
    )
    return image


def apply_stain(image: np.ndarray) -> np.ndarray:
    """
    Apply stain to image with skimage library
    Args:
        image: np.ndarray from remove_background function
    """
    ihc_hed = rgb2hed(image)
    null = np.zeros_like(ihc_hed[:, :, 0])
    h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1),
                          in_range=(0, np.percentile(ihc_hed[:, :, 0], 99)))
    d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1),
                          in_range=(0, np.percentile(ihc_hed[:, :, 2], 99)))
    image = np.dstack((null, d, h))
    return image


def train_augmentation() -> A.Compose:
    """ Train Augmentation """
    transform_train = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.VerticalFlip(p=0.3), # p => 확률을 의미. 0.2는 20%로 적용한다는 뜻
        A.HorizontalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.2),
        ToTensorV2()
    ])
    return transform_train


def inference_augmentation() -> A.Compose:
    """ Inference Augmentation """
    transform_inference = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return transform_inference