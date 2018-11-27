#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
From https://github.com/multi30k/dataset/blob/master/scripts/feature-extractor.
"""

import argparse
import os
from pathlib import Path

import numpy as np

from PIL import Image

from torch.autograd import Variable
import torch.utils.data as data

from torchvision.models import resnet50
from torchvision import transforms

# This script uses the PyTorch's pre-trained ResNet-50 CNN to extract
#   res4f_relu convolutional features of size 1024x14x14
#   avgpool features of size 2048D
# We reproduced ImageNet val set Top1/Top5 accuracy of 76.1/92.8 %
# as reported in the following web page before extracting the features:
#   http://pytorch.org/docs/master/torchvision/models.html
#
# We save the final files as 16-bit floating point tensors to reduce
# the size by 2x. We confirmed that this does not affect the above accuracy.
#
# Organization of the image folder:
#  In order to extract features from an arbitrary set of images,
#  you need to create a folder with a file called `index.txt` in it that
#  lists the filenames of the raw images in an ordered way.
#    -f /path/to/images/train  --> train folder contains 29K images
#                                  and an index.txt with 29K lines.
#


class ImageFolderDataset(data.Dataset):
    """A variant of torchvision.datasets.ImageFolder which drops support for
    target loading, i.e. this only loads images not attached to any other
    label.

    Arguments:
        root (str): The root folder which contains a folder per each split.
        split (str): A subfolder that should exist under ``root`` containing
            images for a specific split.
        resize (int, optional): An optional integer to be given to
            ``torchvision.transforms.Resize``. Default: ``None``.
        crop (int, optional): An optional integer to be given to
            ``torchvision.transforms.CenterCrop``. Default: ``None``.
        indexname (str, optional): The name of the file that contains the list
            of files that should be processed by the CNN. Default: index.txt
    """
    def __init__(self, root, split, resize=None, crop=None, indexname="index.txt"):
        self.split = split
        self.root = Path(root).expanduser().resolve() / self.split
        # Image list in dataset order
        self.index = self.root / indexname
        _transforms = []
        if resize is not None:
            _transforms.append(transforms.Resize(resize))
        if crop is not None:
            _transforms.append(transforms.CenterCrop(crop))
        _transforms.append(transforms.ToTensor())
        _transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(_transforms)

        if not self.index.exists():
            raise(RuntimeError(
                "{} does not exist in {}".format(indexname, self.root)))

        self.image_files = []
        with self.index.open() as f:
            for fname in f:
                fname = self.root / fname.strip()
                assert fname.exists(), "{} does not exist.".format(fname)
                self.image_files.append(str(fname))

    def read_image(self, fname):
        with open(fname, 'rb') as f:
            img = Image.open(f).convert('RGB')
            return self.transform(img)

    def __getitem__(self, idx):
        return self.read_image(self.image_files[idx])

    def __len__(self):
        return len(self.image_files)


def resnet_forward(cnn, x):
    x = cnn.conv1(x)
    x = cnn.bn1(x)
    x = cnn.relu(x)
    x = cnn.maxpool(x)

    x = cnn.layer1(x)
    x = cnn.layer2(x)
    res4f_relu = cnn.layer3(x)
    res5e_relu = cnn.layer4(res4f_relu)

    avgp = cnn.avgpool(res5e_relu)
    avgp = avgp.view(avgp.size(0), -1)
    return res4f_relu, avgp

def extract_img_feats(root, split, batch_size, output_path, savename=None, indexname='index.txt'):
    split_name = split if not savename else savename
    split = split + '2014'
    dataset = ImageFolderDataset(root, split, resize=256, crop=224, indexname=indexname)
    print('Root folder: {} (split: {}) ({} images)'.format(
        root, split, len(dataset)))
    loader = data.DataLoader(dataset, batch_size=batch_size)


    print('Creating CNN instance.')
    cnn = resnet50(pretrained=True)

    # Remove final classifier layer
    del cnn.fc

    # Move to GPU and switch to evaluation mode
    cnn.cuda()
    cnn.train(False)

    # Create placeholders
    # conv_feats = np.zeros((len(dataset), 1024, 14, 14), dtype='float32')
    pool_feats = np.zeros((len(dataset), 2048), dtype='float32')

    n_batches = int(np.ceil(len(dataset) / batch_size))

    bs = batch_size

    for bidx, batch in enumerate(loader):
        x = Variable(batch, volatile=True).cuda()
        res4f, avgpool = resnet_forward(cnn, x)

        pool_feats[bidx * bs: (bidx + 1) * bs] = avgpool.data.cpu()
        # conv_feats[bidx * bs: (bidx + 1) * bs] = res4f.data.cpu()

        print('{:3}/{:3} batches completed.'.format(
            bidx + 1, n_batches), end='\r')
    # Save the files
    output = os.path.join(output_path, '{}-resnet50-avgpool'.format(split_name))
    print("Saving output to {}.npy".format(output))
    np.save(output, pool_feats.astype('float16'))
    # np.save(output + '-res4frelu', conv_feats.astype('float16'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='extract-cnn-features')
    parser.add_argument('-f', '--folder', type=str, required=True,
                        help='Path to MS-COCO.')
    parser.add_argument('-s', '--split', type=str, required=True,
                        help='Folder to image files i.e. /images/train')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='Batch size for forward pass.')
    parser.add_argument('-o', '--output', type=str, default='resnet50',
                        help='Path to output directory.')

    # Parse arguments
    args = parser.parse_args()
    extract_img_feats(args.folder, split, args.batch_size, args.output)
