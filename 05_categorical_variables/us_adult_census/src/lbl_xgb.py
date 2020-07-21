# Label XGBoost

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn import metrics, preprocessing

import config

def run(fold):

    # load the full training data with folds
    df = pd.read_csv(config.FOLDS_FILE)

    # list numerical column
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

    # drop numerical columns
    df = df.drop(num_cols, axis=1)

    # map targets to 0s and 1s
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)

    # -> Define which columns are features
    # All columns are features except id, target and kfold columns
    features = [feature for feature in df.columns if feature not in ("kfold", "income")]

    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesnt matter because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # Label Encode the features
    for col in features:

        # Initialize LabelEncode
        lbl = preprocessing.LabelEncoder()

        # Fit label Encoder
        lbl.fit(df[col])

        # Transform data applying LabelEncoder
        df.loc[:,col] = lbl.transform(df[col])

    # -> Split into Train and Valid dataset
    # training data is where kfold is not equal to provided fold
    # also, not that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # -> Define train and validation dataset values
    x_train = df_train[features].values
    x_valid = df_valid[features].values

    # Initialize logistic regression
    model = xgb.XGBClassifier(n_jobs=-1)

    # fit model on training data (ohe)
    model.fit(x_train, df_train.income.values)

    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {np.round(auc, 4)}")


if __name__ == '__main__':
    for fold in range(5):
        run(fold)
