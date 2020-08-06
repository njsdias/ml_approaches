import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    # read training data
    df = pd.read_csv("../input/imbd.csv")

    # map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    # we create a new column called kfold and fill with -1
    df["kfold"] = -1

    # randomize rows
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch labels
    y = df.sentiment.values

    # initiate the kfold
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new column
    for f, (t_,v_) in enumerate(kf.split(X=df, y=y)):
        df.loc["kfold"] = f

    # save the new csv with kfold column
    df.to_csv("../input/imbd_folds.csv", indesx=False)
