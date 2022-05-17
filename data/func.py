import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn

from skimage import io, transform


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        h, w = image.shape[:2]
        # if h > w:
        #     new_h, new_w = self.output_size * h / w, self.output_size
        # else:
        #     new_h, new_w = self.output_size, self.output_size * w / h

        # new_h, new_w = int(new_h), int(new_w)
        new_h, new_w = self.new_h, self.new_w

        new_image = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        labels = []

        for i in target:
            i["base"][:, 0] *= [new_w / w, new_h / h]
            i["base"][:, 1] *= [new_w / w, new_h / h]
            i["base"][:, 2] *= [new_w / w, new_h / h]
            i["base"][:, 3] *= [new_w / w, new_h / h]
            new_label = np.squeeze(i["base"])
            if i["type"] == "Car":
                new_label = np.append(new_label, 0)
            else:
                new_label = np.append(new_label, 1)
            labels.append(new_label)
        labels = np.asarray(labels)

        return {'image': torch.from_numpy(new_image), 'labels': torch.from_numpy(labels)}

class Cropper(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        h, w = image.shape[:2]
        # if h > w:
        #     new_h, new_w = self.output_size * h / w, self.output_size
        # else:
        #     new_h, new_w = self.output_size, self.output_size * w / h

        # new_h, new_w = int(new_h), int(new_w)
        new_h, new_w = self.new_h, self.new_w

        new_image = image[0:new_h, 0:new_w, :]

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        labels = []

        for i in target:
            new_label = np.squeeze(i["base"])
            if i["type"] == "Car":
                new_label = np.append(new_label, 0)
            else:
                new_label = np.append(new_label, 1)
            labels.append(new_label)
        labels = np.asarray(labels)

        return {'image': torch.from_numpy(new_image), 'labels': torch.from_numpy(labels)}


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        return {'image': ((image / 255.0 - self.mean) / self.std), 'labels': labels}


def collater(data):
    imgs = [s['image'] for s in data]
    annots = [s['labels'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 9)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 9)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'image': padded_imgs, 'labels': annot_padded}
