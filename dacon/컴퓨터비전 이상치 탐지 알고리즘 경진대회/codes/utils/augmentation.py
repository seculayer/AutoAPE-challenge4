import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
    A.RandomGamma(gamma_limit=(90, 110)),
    A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10),
    A.Transpose(),
    A.RandomRotate90(),
    A.OneOf([A.NoOp(), A.MultiplicativeNoise(), A.GaussNoise(), A.ISONoise()]),
    A.OneOf(
        [
            A.NoOp(p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10)
        ],
        p=0.2,
    ),
    A.OneOf([A.ElasticTransform(), A.GridDistortion(), A.NoOp()]),
    ToTensorV2(),
])


test_transform = A.Compose([
    ToTensorV2()
])