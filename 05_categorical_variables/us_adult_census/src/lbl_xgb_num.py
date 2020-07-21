# XGBoost with numerical features applying LabelEncoder to categorical features

import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn import metrics, preprocessing

import config

def run(fold):

    # load the full training data with folds
    df = pd.read_csv(config.FOLDS_FILE)

    # list of numerical columns
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]
    # map targets to 0s and 1s
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)

    # all columns are features except kfold & income columns
    features = [
        feature for feature in df.columns if feature not in ("kfold", "income")
    ]

    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesnt matter because all are categories
    for col in features:
        # do not encode the numerical columns
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")

    for col in features:
        if col not in num_cols:
            # initialize LabelEnconder for each feature
            lbl = preprocessing.LabelEncoder()

            # fit label encoder on all data
            lbl.fit(df[col])

            # transform data
            df.loc[:, col] = lbl.transform(df[col])

    # get training data using kfolds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using kfolds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values

    # get validation data
    x_valid = df_valid[features].values

    # initialize xgboost model
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
    print(f"Fold = {fold}, AUC = {np.round(auc,4)}")


if __name__ == "__main__":
    for fold in range(5):
        run(fold)
