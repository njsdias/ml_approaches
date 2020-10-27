
# Find optimal value(weight) for the Weight Rank Averaging
# Using an Optimization function


import glob
import pandas as pd
import numpy as np

from sklearn import metrics

from functools import partial
from scipy.optimize import fmin


class OptimizeAUC:
    def __init__(self):
        self.coef_ = 0

    def _auc(self, coef, X, y):
        # we have three coefficients.
        # One weight for each model
        x_coef = X * coef
        predictions = np.sum(x_coef, axis=1)
        auc_score = metrics.roc_auc_score(y, predictions)

        # we are minimize the function.
        # So we need to return the negative value of auc_score
        return -1.0 * auc_score

    def fit(self, X, y):
        """
        :param X: train data
        :param y: targets
        :return:
        """
        # take X and y and returns auc
        partial_loss = partial(self._auc, X=X, y=y)

        # make a initial values for the coefficients to
        # start the optimization process
        init_coef = np.random.dirichlet(np.ones(X.shape[1]))

        # evaluate the coefficients
        self.coef_ = fmin(partial_loss, init_coef, disp=True)

    def predict(self, X):

        # multiply each coeficient with the train dataset
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)

        return predictions


def run_training(pred_df, fold):

    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[["lr_pred", "lr_cnt_pred", "rf_svd_pred"]].values
    xvalid = valid_df[["lr_pred", "lr_cnt_pred", "rf_svd_pred"]].values

    opt = OptimizeAUC()
    opt.fit(xtrain, train_df.sentiment.values)
    preds = opt.predict(xvalid)

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

    wt_avg = coefs[0] * df.lr_pred.values + coefs[1] * df.lr_cnt_pred.values + coefs[2] * df.rf_svd_pred.values

    print("optimal auc after finding coefs")
    print(metrics.roc_auc_score(targets, wt_avg))

    #print(metrics.roc_auc_score(preds_df.sentiment.values, preds_df.opt_pred.values))

