import create_folds
from pathlib import Path
import config

import joblib
import pandas as pd
import numpy as np

from sklearn import metrics

import argparse
import os


import model_dispatcher


def run(fold, model):
    # read training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # training data is where kfold is not equal to provided fold
    # also, note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column from dataframe and convert it to
    # a numpy array by using .values.
    # target is label column in the dataframe
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values

    # similarly, for validation, se have
    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    # classifier
    # initialize simple decision tree classifier from sklearn
    # clf = tree.DecisionTreeClassifier()
    clf = model_dispatcher.models[model]

    # fit the model on training data
    clf.fit(x_train, y_train)

    # create predictions
    preds = clf.predict(x_valid)

    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={np.round(accuracy,4)}")

    # save the model
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )


if __name__== "__main__":
    my_file = Path(config.TRAINING_FILE)
    if not my_file.is_file():
        create_folds()

    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their type
    # currently, we only need fold
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )

    # read the arguments from the command line
    args = parser.parse_args()

    # run the fold specified by command line arguments
    run(
        fold=args.fold,
        model=args.model
        
    )
