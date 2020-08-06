import os
import sys
import torch

import numpy as np
import pandas as pd

import torch.nn as nn

import segmentation_models_pythorch as smp

from apex import amp
from collections import OrderedDict
from sklearn import model_selection
from tqdm import tqdm
from torch.optim import lr_scheduler

from dataset import SIIMDataset

import tqdm as tqdm

import config


def train(dataset, data_loader, model, criterion, optimizer):
    """
    training function that trains for one epoch
    :param dataset: dataset class (SIIMDataset)
    :param data_loader: torch dataset loader
    :param model: model
    :param criterion: loss function
    :param optimizer: adam, sgd, etc.
    """
    # put the model in train mode
    model.train()

    # calculate the number of batches
    num_batches = int(len(dataset) / data_loader.batch_size)

    # init tqdm to track progress
    tk0 = tqdm(data_loader, total=num_batches)

    # loop over all batches
    for data in tk0:
        # fetch input images and masks
        # from dataset batch
        inputs = data["images"]
        targets = data["mask"]

        # move images and masks to cpu/gpu device
        inputs = inputs.to(config.DEVICE, dtype=torch.float)
        targets = targets.to(config.DEVICE, dtype=torch.float)

        #zero grad the optimzer
        optimizer.zero_grad()

        # forward step of model
        outputs = model(inputs)

        # calculate loss
        loss = criterion(outputs, targets)

        # backward loss is calculated on a scaled loss
        # context since we are using mixed precision training
        # if you are not using mixed precision training,
        # you can use loss.backward() and delete the following
        # two lines of code
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
            # step the optimizer
            optimizer.step()

    # close tqdm
    tk0.close()


def evaluate(dataset, data_loader, criterion, model):
    """
    training function that trains for one epoch
    :param dataset: dataset class (SIIMDataset)
    :param data_loader: torch dataset loader
    :param model: model
    :param criterion: loss function
    :param optimizer: adam, sgd, etc.
    """
    # put the model in train mode
    model.eval()

    # calculate the number of batches
    num_batches = int(len(dataset) / data_loader.batch_size)

    # init tqdm to track progress
    tk0 = tqdm(data_loader, total=num_batches)

    # initial final_loss to 0
    final_loss = 0

    # we need no_grad context of torch. this save memory
    with torch.no_grad():
        # loop over all batches
        for data in tk0:
            # fetch input images and masks
            # from dataset batch
            inputs = data["images"]
            targets = data["mask"]

            # move images and masks to cpu/gpu device
            inputs = inputs.to(config.DEVICE, dtype=torch.float)
            targets = targets.to(config.DEVICE, dtype=torch.float)

            # forward step of model
            output = model(inputs)

            loss = criterion(output, targets)
            # add loss to final loss
            final_loss += loss

    # close tqdm
    tk0.close()

    # return average loss over all batches
    return final_loss / num_batches


if __name__ == "__main__":

    # read the training csv file
    df = pd.read_csv(config.TRAINING_CSV)

    # split data into training and validation
    df_train, df_valid = model_selection.train_test_split(df, random_state=42, test_size=0.1)

    # training and validation images lists/arrays
    training_images = df_train.image_id.values
    validation_images = df_valid.image_id.values

    # fetch unet model from segmentation models
    # with specified encoder architecture
    model = smp.Unet(encoder_name=config.ENCODER,
                     encoder_weights=config.ENCODER_WEIGHTS,
                     classes=1,
                     activation=None,)

    # segmentation model provides you with a pre-processing
    # function that can be used for normalizing images
    # normalization is only applied on images and not masks
    prep_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS)

    # send model to device
    model.to(config.DEVICE)

    ##
    # init training dataset
    # transform is True for training data
    train_dataset = SIIMDataset(training_images,
                                transform=True,
                                preprocessing_fn=prep_fn,)

    # wrap training dataset in torch's dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.TRAINING_BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=12)

    ##
    # init validation dataset
    # augmentations is disabled
    valid_dataset = SIIMDataset(validation_images,
                                transform=False,
                                preprocessing_fn=prep_fn,)

    # wrap validation dataset in torch's dataloader
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=config.TEST_BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=4)

    # NOTE: define the criterion here
    # this is left as an excercise
    # code won't work without defining this
    # criterion = ......

    # we will use Adam optimizer for faster convergence
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # reduce learning rate when we reach a plateau on loss
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               mode="min",
                                               patience=3,
                                               verbose=True)

    # wrap model and optimizer with NVIDIA's apex
    # this is used for mixed precision training
    # if you have a GPU that supports mixed precision,
    # this is very helpful as it will allow us to fit larger images
    # and larger batches
    model, optimizer = amp.initialize(model,
                                      optimizer,
                                      opt_level="O1",
                                      verbosity=0)

    # if we have more than one GPU, we can use both of them!
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")

    model = nn.DataParallel(model)

    # some logging
    print(f"Training batch size: {config.TRAINING_BATCH_SIZE}")
    print(f"Test batch size: {config.TEST_BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Image size: {config.IMAGE_SIZE}")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(valid_dataset)}")
    print(f"Encoder: {config.ENCODER}")

    # loop over all epochs
    for epoch in range(config.EPOCHS):
        print(f"Training Epoch: {epoch}")
        # train for one epoch
        train(train_dataset, train_loader, model, criterion, optimizer)

        print(f"Validation Epoch: {epoch}")
        # calculate validation loss
        val_log = evaluate(valid_dataset, valid_loader, criterion, model)

        # step the scheduler
        scheduler.step(val_log["loss"])
        print("\n")




