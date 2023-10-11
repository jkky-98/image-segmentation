##################libary import####################
import os
import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn.functional as F
import random
import sys
# 본인 segmentation_models_pytorch 폴더의 경로를 입력하세요
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smpu
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
from torch.utils.data import Subset
# 위에 경로 잘 설정 부탁드립니다 ㅎㅎ
import ssl

##################module setting#####################

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose, Lambda
    )
def get_training_augmentation(width=320, height=320):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
        A.RandomCrop(height=height, width=width, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        # A.Normalize(mean=[87.24029665, 91.22533398, 82.92776534],
        #             std=[49.33672613, 44.09863433, 42.24505498])
    ]
    return A.Compose(train_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        Lambda(image=preprocessing_fn), # preprocessing with encoder contained encoder weights
        Lambda(image=to_tensor, mask=to_tensor) # ability to tensor
    ]
    return Compose(_transform)

class SatelliteDatasetInfer(Dataset):
    def __init__(self, csv_file, transforms=None, preprocessing=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        img = Image.open(img_path).convert('RGB') # PIL Image (not numpy dtype)

        if self.infer: # in infer mode, only give image
            if self.transforms:
              img = self.transforms(image=np.array(img))['image']
            if self.preprocessing:
              img = self.preprocessing(image=img)['image']
            return img

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (1024, 1024, 1)) # shape : (1024, 1024, 1), numpy dtype

        if self.transforms:
            augmented = self.transforms(image=np.array(img), mask=mask) # transforming : plz insert input(numpy dtype)
            img = augmented['image']
            mask = augmented['mask']

        if self.preprocessing:
            augmented = self.preprocessing(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img, mask #

def numerical_sort(value):
    # 정렬 시, 파일 이름의 숫자 부분을 기준으로 정렬하기 위해 사용
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

class CropDataset(Dataset):
    def __init__(self, img_path, mask_path, transforms=None, preprocessing=None, infer=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.img_lst = []
        self.mask_lst = []
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.infer = infer
        self.sort_by_numerical()

    def sort_by_numerical(self):
        self.img_lst = os.listdir(self.img_path)
        self.img_lst.sort(key=self.numerical_sort)
        self.mask_lst = os.listdir(self.mask_path)
        self.mask_lst.sort(key=self.numerical_sort)

    def numerical_sort(self, value):
        # 정렬 시, 파일 이름의 숫자 부분을 기준으로 정렬하기 위해 사용
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, idx):
        img_pth = self.img_path +'/'+ self.img_lst[idx]
        mask_pth = self.mask_path +'/'+ self.mask_lst[idx]

        img = Image.open(img_pth).convert('RGB') # PIL Image (not numpy dtype)

        if self.infer: # in infer mode, only give image
            if self.transforms:
                img = self.transforms(image=np.array(img))['image']
            if self.preprocessing:
                img = self.preprocessing(image=img)['image']
            return img
        mask = cv2.imread(mask_pth, cv2.IMREAD_GRAYSCALE)
        mask = np.reshape(mask, (256, 256, 1))
        mask = mask / 255
         # shape : (1024, 1024, 1), numpy dtype
        if self.transforms:
            augmented = self.transforms(image=np.array(img), mask=mask) # transforming : plz insert input(numpy dtype)
            img = augmented['image']
            mask = augmented['mask']

        if self.preprocessing:
            augmented = self.preprocessing(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img, mask

def split_dataset(dataset, split_ratio=0.8, random_seed=None):
    """
    데이터셋을 훈련과 검증 데이터셋으로 나누는 함수

    Args:
        dataset (torch.utils.data.Dataset): 전체 데이터셋
        split_ratio (float): 훈련 데이터셋의 비율 (0과 1 사이의 값)
        random_seed (int): 랜덤 시드 값 (기본값: None)

    Returns:
        torch.utils.data.Dataset: 훈련 데이터셋
        torch.utils.data.Dataset: 검증 데이터셋
    """

    # 데이터셋 크기를 구하고 셔플합니다.
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(indices)

    # 지정한 비율로 데이터셋을 나눕니다.
    split = int(split_ratio * dataset_size)
    train_indices, val_indices = indices[:split], indices[split:]

    # Subset으로 나눈 후 반환합니다.
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset

def get_test_augmentation(width=224, height=224):
    train_transform = [
        A.GaussNoise(p=0),
        # A.OneOf(
        #     [
        #         A.CLAHE(p=1),
        #         A.RandomBrightnessContrast(p=1),
        #         A.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),

        # A.OneOf(
        #     [
        #         A.Sharpen(p=1),
        #         A.Blur(blur_limit=3, p=1),
        #         A.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),

        # A.OneOf(
        #     [
        #         A.RandomBrightnessContrast(p=1),
        #         A.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
        # A.Normalize(mean=[87.24029665, 91.22533398, 82.92776534],
        #             std=[49.33672613, 44.09863433, 42.24505498])
    ]
    return A.Compose(train_transform)

def iscuda():
    if torch.cuda.is_available():
        # if colab notebook can do in CUDA
        device = torch.device("cuda")
        print("You have a GPU with CUDA enabled.")
        print("GPU in use:", torch.cuda.get_device_name(0))
    else:
        # if colab notebook can't do in CUDA
        device = torch.device("cpu")
        print("Cuda currently cannot use GPU, I will use CPU")

def set_workspace(path):
  desired_directory = path
  os.chdir(desired_directory)
  current_directory = os.getcwd()
  print("Changed work path :", current_directory)

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def main():
    seed_torch()

    ####################현재 GPU설정 되었는지 확인###################
    iscuda()
    device = torch.device("cuda")

    # SSL 인증서 검증 비활성화
    ssl._create_default_https_context = ssl._create_default_https_context = ssl._create_unverified_context

    ENCODER = 'resnet101'  # fix
    ENCODER_WEIGHTS = 'imagenet'  # fix
    CLASSES = ['building']  # fix
    ACTIVATION = 'sigmoid'  # fix
    # create segmentation model with pretrained encoder

    # model = smp.DeepLabV3Plus(
    #     encoder_name=ENCODER,
    #     encoder_weights=ENCODER_WEIGHTS,
    #     classes=len(CLASSES),
    #     activation=ACTIVATION,
    # )

    ############################################모델 불러오기############################################################
    model = torch.load('C:/Users/JKKY/Desktop/project_dacon/deeplabv3+/dacon/best_model_DLV3+_se_resnet101_2_256_19+.pth ')
    ##################################################################################################################

    # encoder setting with encoder weights
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    _preprocessing = get_preprocessing(preprocessing_fn)

    # base train dataset generate
    dataset = CropDataset(img_path='C:/Users/JKKY/Desktop/project_dacon/deeplabv3+/dacon/train_img_16crop_255',
                          mask_path='C:/Users/JKKY/Desktop/project_dacon/deeplabv3+/dacon/mask_img_16crop_255255',
                          transforms=get_training_augmentation(256, 256),
                          preprocessing=_preprocessing,
                          infer=False)
    # sampler generate
    train_dataset, val_dataset = split_dataset(dataset, 0.9, 42)

    # train % validation DataLoader generate

    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=2, pin_memory=True, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2, pin_memory=True, drop_last=True, shuffle=False)

    # base test dataset genrate
    test_dataset = SatelliteDatasetInfer(csv_file='./test.csv',
                                         transforms=get_training_augmentation(224, 224)
                                         , preprocessing=_preprocessing
                                         , infer=True)

    # test DataLoader generate
    batch_size_test = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, num_workers=2, pin_memory=True)

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.000004),
    ])

    train_epoch = smpu.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smpu.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    EPOCHS = 10

    max_score = 0

    # train accurascy, train loss, val_accuracy, val_loss
    x_epoch_data = []
    train_dice_loss = []
    train_iou_score = []
    valid_dice_loss = []
    valid_iou_score = []

    for i in range(EPOCHS):

        print(f'\nEpoch: {i + 1}')
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        x_epoch_data.append(i)
        train_dice_loss.append(train_logs['dice_loss'])
        train_iou_score.append(train_logs['iou_score'])
        valid_dice_loss.append(valid_logs['dice_loss'])
        valid_iou_score.append(valid_logs['iou_score'])

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model_DLV3+_se_resnet101_2_256_20+.pth')
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

if __name__ == "__main__":
	main()