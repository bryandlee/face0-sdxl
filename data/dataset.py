import json
from torch.utils.data import Dataset, ConcatDataset
from typing import List, Dict

import albumentations as A
import numpy as np
import torch

from common.util import Registerable

from .preprocess import (
    open_image,
    make_image_face_pair,
)


# TODO: load captions & tokenize


class BaseDataset(Dataset, Registerable):
    pass


class ConcatDataset(BaseDataset, ConcatDataset):
    def __init__(self, datasets: List[BaseDataset | Dict]):
        _datasets = []
        for dataset in datasets:
            if isinstance(dataset, dict):
                dataset = build_dataset(**dataset)
            _datasets.append(dataset)
        super().__init__(_datasets)


class FaceDetectionDatasetJson(BaseDataset):
    def __init__(
        self,
        path: str,
        use_mask: bool = False,
        augment: bool = True,
        multiple: int = 1,
        **face_crop_kwargs,
    ):
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())
        self.num_images = len(self.image_paths)
        self.multiple = multiple

        self.use_mask = use_mask
        self.face_crop_kwargs = face_crop_kwargs

        face_augmentation = [A.HorizontalFlip(p=0.5)]
        if augment:
            face_augmentation += [
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=1.0),
                        A.RandomGamma(p=1.0),
                    ],
                    p=0.2,
                ),
            ]
        self.face_augmentation = A.Compose(face_augmentation)

        image_augmentation = [A.HorizontalFlip(p=0.5)]
        if augment:
            image_augmentation += [
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=1.0),
                        A.RandomGamma(p=1.0),
                    ],
                    p=0.2,
                ),
                A.Rotate(limit=20, p=0.5),
            ]
        self.image_augmentation = A.Compose(image_augmentation)

    def __len__(self):
        return self.num_images * self.multiple

    def __getitem__(self, idx):
        image_path = self.image_paths[idx % self.num_images]
        image = open_image(image_path)
        image, face, mask = make_image_face_pair(
            image=image,
            face_ccwh=self.annotations[image_path]["detections"][0]["bbox"],
            face_keypoint5=self.annotations[image_path]["detections"][0].get("kps", None),
            **self.face_crop_kwargs,
            get_mask=self.use_mask,
        )

        data = {"image": image}
        if self.use_mask:
            data["mask"] = mask
        data = self.image_augmentation(**data)

        data["face"] = self.face_augmentation(image=face)["image"]
        data["image_path"] = image_path

        return data


def build_dataset(_target_: str, **kwargs):
    return BaseDataset.registry[_target_](**kwargs)


def images_to_4d_tensor(images: List[np.ndarray]) -> torch.Tensor:
    # list -> np.ndarray (BHWC) -> BCHW
    if isinstance(images[0], np.ndarray) and len(images) == 1:
        data = images[0][np.newaxis]
    else:
        data = np.stack(images)

    data = data.transpose(0, 3, 1, 2)
    data = (data.astype(np.float32) - 127.5) * 0.00784313725
    data = torch.from_numpy(data).contiguous()
    return data


def masks_to_3d_tensor(images: List[np.ndarray]) -> torch.Tensor:
    if isinstance(images[0], np.ndarray) and len(images) == 1:
        data = images[0][np.newaxis]
    else:
        data = np.stack(images)
    data = torch.from_numpy(data.astype(np.float32)).contiguous()
    return data


def collate(batch: List[Dict]):
    collated_batch = {}
    for key in batch[0]:
        collated = [data[key] for data in batch]
        if key in ["image", "face"]:
            collated = images_to_4d_tensor(collated)
        elif key in ["mask"]:
            collated = masks_to_3d_tensor(collated)
        collated_batch[key] = collated
    return collated_batch


def get_data_iter(loader):
    while True:
        for batch in loader:
            yield batch
