import os
import glob
import torch
import numpy as np
from sklearn import preprocessing, model_selection, metrics

import dataset

import sys
sys.path.append("../src/")
import config

# this two lines is related with model.py
import engine
from model import CaptchaModel

from pprint import pprint


def decode_predictions(preds,encoder):
    # batch size timestamps predictions
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()

    # a list to save captcha predictions
    cap_preds = []
    # goes into each every image in every batch
    for j in range(preds.shape[0]):
        temp = []  # temporary list
        # goes to every value predicted
        for k in preds[j, :]:
            # subtract one values because we added one
            # so respecting the number of values
            k = k - 1
            if k == -1:
                temp.append("*")  # * it is our unknown value
            else:
                temp.append(encoder.inverse_transform([k])[0])
        temporary_predictions = "".join(temp)
        cap_preds.append(temporary_predictions)

    return cap_preds


def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.png"))

    # "/../../sdfrt.png" the next line only select the name of the file: sdfrt
    targets_orig = [x.split("/")[-1][:-4] for x in image_files]

    # sdfrt -> [s, d, f, r, t]
    targets = [[c for c in x] for x in targets_orig]

    targets_flat = [c for clist in targets for c in clist]

    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)

    # Encode the targets
    targets_enc = [lbl_enc.transform(x) for x in targets]

    # Transform targets_enc to np.array
    # The labels are encoded from 0 to N-1 where N is the number of labels
    # we want to keep 0 to unknown so add 1
    targets_enc = np.array(targets_enc) + 1

    print(targets)
    print(np.unique(targets_flat))
    print(targets_enc)
    print(len(lbl_enc.classes_))

    # split in train, test for: imgs, targets, orig_targets
    train_imgs, test_imgs, train_targets, test_targets, _, test_orig_targets = \
        model_selection.train_test_split(image_files,
                                         targets_enc, targets_orig,
                                         test_size=0.1, random_state=42)

    train_dataset = dataset.ClassificationDataset(image_paths=train_imgs, targets=train_targets,
                                                  resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )

    test_dataset = dataset.ClassificationDataset(image_paths=test_imgs, targets=test_targets,
                                                 resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    model = CaptchaModel(num_chars=len(lbl_enc.classes_))
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        valid_preds, valid_loss = engine.train_fn(model, test_loader)

        valid_cap_preds = []
        for valid_pred in valid_preds:
            current_preds = decode_predictions(valid_pred, lbl_enc)
            valid_cap_preds.extend(current_preds)

        pprint(list(zip(test_orig_targets, valid_cap_preds))[6:10])
        print(f"Epoch: {epoch}, train_loss={train_loss}, valid_loss={valid_loss}")


if __name__ == "__main__":
    run_training()
    # In every step we are predicting 75 values for each image that can
    # have unknown values(represented here by the symbol*)
    # for instance: *sdsf***gfdg***6655**687ghj**hgkh**ukn**nmj***hdsdcghnhkgf****hhjumi*****etr
    # We need to remove these unknown values
    # For to see the result related with predictions we need to decode it (decode_prediction)

