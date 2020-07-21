import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, ensemble


def run(fold):
    # load full training data with folds
    df = pd.read_csv("../input/cat_train_folds.csv")

    # -> Define which columns are features
    # All columns are features except id, target and kfold columns
    features = [
        feature for feature in df.columns if feature not in ("id", "target", "kfold")]

    # -> Fill all NaN values with NONE
    # The fact all columns are going converted to "strings"
    # it doesn’t matter because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # Label encode the features
    for col in features:
        # initialize LabelEncoder
        lbl = preprocessing.LabelEncoder()

        # fit label encoder to all data
        lbl.fit(df[col])

        # Transform all the data
        df.loc[:, col] = lbl.transform(df[col])

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
    model = ensemble.RandomForestClassifier(n_jobs=-1)

    # Fit model
    model.fit(x_train, df_train.target.values)

    # Predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_train.target.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {np.round(auc, 4)}")


if __name__ == "__main":
    for fold_ in range(5):
        run(fold_)
