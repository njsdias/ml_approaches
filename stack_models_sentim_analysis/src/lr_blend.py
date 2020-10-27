
import glob
import pandas as pd
import numpy as np

from sklearn import metrics

from functools import partial
from scipy.optimize import fmin

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression




def run_training(pred_df, fold):

    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[["lr_pred", "lr_cnt_pred", "rf_svd_pred"]].values
    xvalid = valid_df[["lr_pred", "lr_cnt_pred", "rf_svd_pred"]].values

    scl = StandardScaler()
    xtrain = scl.fit_transform(xtrain)
    xvalid = scl.transform(xvalid)

    opt = LogisticRegression()
    opt.fit(xtrain, train_df.sentiment.values)
    preds = opt.predict_proba(xvalid)[:, 1]

    auc = metrics.roc_auc_score(valid_df.sentiment.values, preds)
    print(f"{fold}, {auc}")

    #valid_df .loc[:, "opt_pred"] = preds

    return opt.coef_


if __name__ == "__main__":
    # load file names in memory
    files = glob.glob("../model_preds/*.csv")

    # for the first iteration
    df = None

    for f in files:
        if df is None:
            # save file info in dataframe df
            df = pd.read_csv(f)
        else:
            # save file info in a temporary dataframe
            temp_df = pd.read_csv(f)
            # merge the new temp df with the new one using "id" as key
            df = df.merge(temp_df, on='id', how='left')

    targets = df.sentiment.values
    pred_cols = ["lr_pred", "lr_cnt_pred", "rf_svd_pred"]

    #preds_df = []
    coefs = []
    for j in range(5):
        # preds.append(run_training(df, j))
        coefs.append(run_training(df, j))

    # preds_df = pd.concat(preds_df)
    coefs = np.array(coefs)
    print(coefs)
    coefs = np.mean(coefs, axis = 0)
    print(coefs)

    wt_avg = coefs[0][0] * df.lr_pred.values + \
             coefs[0][1] * df.lr_cnt_pred.values +\
             coefs[0][2] * df.rf_svd_pred.values

    print("optimal auc after finding coefs")
    print(metrics.roc_auc_score(targets, wt_avg))

    #print(metrics.roc_auc_score(preds_df.sentiment.values, preds_df.opt_pred.values))

