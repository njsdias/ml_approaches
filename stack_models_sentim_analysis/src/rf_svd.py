# Random Forest with Singular Vector Decomposition

import pandas as pd


from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier

import joblib


def run_training(fold):
    # read the train folds geenrated by create_folds.py
    df = pd.read_csv("../input/train_folds.csv")

    # to guarantee the review texts are understanding as strings
    df.review = df.review.apply(str)

    # create train dataset using df
    # note the train datataset is all data that is not belong
    # to the fold number label. If the number of fold is zero
    # the train will be other folders different to zero
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # create test datset
    # it will be the data where kfold as the number of the fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # convert text into number (vectorized) using tf-idf
    tfv = TfidfVectorizer()
    tfv.fit(df_train.review.values)

    # apply tf-idf to vectorized the text reviews
    # in train and validation dataset
    xtrain = tfv.transform(df_train.review.values)
    xvalid = tfv.transform(df_valid.review.values)

    # Initialize Singular Value Decomposition
    # to promote the dimensionality reduction
    svd = decomposition.TruncatedSVD(n_components=120)
    svd.fit(xtrain)

    xtrain_svd = svd.transform(xtrain)
    xvalid_svd = svd.transform(xvalid)

    # select labels using train and validation dataset
    ytrain = df_train.sentiment.values
    yvalid = df_valid.sentiment.values

    # Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    # model train dataet using Logistic Regression
    clf.fit(xtrain_svd, ytrain)

    # save model
    filename = "../model_preds/rf_svd__model.sav"
    joblib.dump(clf, filename)

    # Calculate Predictions using the model with
    # validation dataset
    pred = clf.predict_proba(xvalid_svd)[:, 1]

    # calculate AUC
    auc = metrics.roc_auc_score(yvalid, pred)
    print(f"fold={fold}, auc={auc}")

    # store predictions in a new column
    df_valid.loc[:, "rf_svd_pred"] = pred

    return df_valid[["id", "sentiment", "kfold", "rf_svd_pred"]]


if __name__ == "__main__":
    # list of dataframes
    dfs = []
    for j in range(5):
        temp_df = run_training(j)
        dfs.append(temp_df)

    # save all predictions in one dataframe
    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)
    fin_valid_df.to_csv("../model_preds/rf_svd.csv", index=False)

