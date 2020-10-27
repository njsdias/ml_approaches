# Logistic regression using Count Vectorized

import pandas as pd

from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

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
    tfv = CountVectorizer()
    tfv.fit(df_train.review.values)

    # apply tf-idf to vectorized the text reviews
    # in train and validation dataset
    xtrain = tfv.transform(df_train.review.values)
    xvalid = tfv.transform(df_valid.review.values)

    # select labels using train and validation dataset
    ytrain = df_train.sentiment.values
    yvalid = df_valid.sentiment.values

    # Logistic Regression Classifier
    clf = linear_model.LogisticRegression()

    # model train dataet using Logistic Regression
    clf.fit(xtrain, ytrain)

    # save model
    filename = "../model_preds/lr_cnt_model.sav"
    joblib.dump(clf, filename)

    # Calculate Predictions using the model with
    # validation dataset
    pred = clf.predict_proba(xvalid)[:, 1]

    # calculate AUC
    auc = metrics.roc_auc_score(yvalid, pred)
    print(f"fold={fold}, auc={auc}")

    # store predictions in a new column
    df_valid.loc[:, "lr_cnt_pred"] = pred

    return df_valid[["id", "sentiment", "kfold", "lr_cnt_pred"]]


if __name__ == "__main__":
    #list of dataframes
    dfs = []
    for j in range(5):
        temp_df = run_training(j)
        dfs.append(temp_df)

    # save all predictions in one dataframe
    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)
    fin_valid_df.to_csv("../model_preds/lr_cnt.csv", index=False)














