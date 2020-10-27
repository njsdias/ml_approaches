import glob
import pandas as pd
import numpy as np

from sklearn import metrics

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

    print(df.head())


    targets = df.sentiment.values

    pred_cols = ["lr_pred", "lr_cnt_pred", "rf_svd_pred"]

    # Calculate the Overall AUC
    for col in pred_cols:
        auc = metrics.roc_auc_score(targets, df[col].values)
        print(f"{col}, overall_auc={auc}")

    # Print the average AUC
    print("average")
    avg_pred = np.mean(df[["lr_pred", "lr_cnt_pred", "rf_svd_pred"]].values, axis=1)
    print(metrics.roc_auc_score(targets, avg_pred))

    print("weighted average")
    lr_pred = df.lr_pred.values
    lr_cnt_pred= df.lr_cnt_pred.values
    rf_svd_pred = df.rf_svd_pred.values

    avg_pred = (lr_pred + 3 * lr_cnt_pred + rf_svd_pred) / 5
    print(metrics.roc_auc_score(targets, avg_pred))

    # Note: If we working with value prediction 0 or 1 we need to
    # choose correctly as instance:
    # lr_pred = 0; lr_cnt_pred=1; rf_svd_pred=1
    # final result will be 1
    # But in this example we are looking for probabilities

    print("rank averaging")
    lr_pred = df.lr_pred.rank().values
    lr_cnt_pred= df.lr_cnt_pred.rank().values
    rf_svd_pred = df.rf_svd_pred.rank().values

    avg_pred = (lr_pred + lr_cnt_pred + rf_svd_pred) / 3
    print(metrics.roc_auc_score(targets, avg_pred))


    print("weighted rank averaging")
    lr_pred = df.lr_pred.rank().values
    lr_cnt_pred= df.lr_cnt_pred.rank().values
    rf_svd_pred = df.rf_svd_pred.rank().values

    avg_pred = (lr_pred + 3 * lr_cnt_pred + rf_svd_pred) / 5
    print(metrics.roc_auc_score(targets, avg_pred))






