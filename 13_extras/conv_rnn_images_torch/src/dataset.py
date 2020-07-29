# we need the DataLoader

import albumentations
import torch
import numpy as np

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassificationDataset:
    def __init__(self, image_paths, targets, resize=None):
        # resize: tuple  (height, width)
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.aug = albumentations.Compose([albumentations.Normalize(always_apply=True)])

    def __len__(self):
        return len(self.image_paths)

    def _getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        targets = self.targets[item]

        if self.resize is not None:
            # resize receives a tuple (width, height)
            # do an image resample
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

        # convert image to numpy array
        image = np.array(image)

        # apply augumentation to images
        augumented = self.aug(image=image)
        image = augumented["image"]

        # transpose to channel first
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # return a dictionary where we convert images and targets to torch tensor
        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }

