# Description

Here is showed how is implemented the **U-Net architecture**.
The "U" in the named is related with the shape of the network.

This architecture is used in image segmentationas you can in the
 title of the paper: Convolutional Networks for Biomedical Image Segmentation".

You can find the original paper here:

https://arxiv.org/pdf/1505.04597.pdf


Thus, we can use this architecture for the pneumothorax problem that requires to segment 
pneumothorax.

# Implementation

In the link below you can find a more detailed explanation about this architecture: 
 
https://www.youtube.com/watch?v=u1loyDCoGbE

Note, this is the original implementation of U-Net architecture. We have available
a more sophisticated U-Net version implementation in PyTorch.

The encoder part of the U-Net is a simple convolutional network. Thus we can 
replaced it with ResNet which is a based encoder pre-trained on ImageNet
 and a generic decoder.
 
 
# Segmentation Problems
Most of the problems has two inputs: original image and a mask.
Here we are using RLE (Run-length encoding). 
Bit-level RLE schemes encode runs of multiple bits in a scan line and ignore byte and word boundaries. 

# Files Description 

In `simple_unet.py` you can find the U-Net architecture implementation.

In `config.py` you find the main model configurations.

In `dataset.py` you load our csv file and input images and the output will be outputs images and
mask images. Thus in our `input` folder we will have a csv file consisting only 
of image ids which are also filenames. 

Once, we have the dataset class, we can create a `train.py`.
In this file we are using the PyTorch U-Net implementation and not the one in the `simple_unet`
We are training using  NVIDIA apex that is available natively in PyTorch
from version 1.6.0+.

In `plant.py` we can find a multi-calss classification implemented using _Well That’s Fantastic Machine Learning_ package. This file
build a multi-class classification model for plant images from the plant pathology
challenge of FGVC 2020. The objective of this problem is given pictures of apple leaves, distinguish between:
- leaves which are healthy, 
- those which are infected with apple rust, 
- those that have apple scab, 
- and those with more than one disease

The dataset can be found here:

https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data



