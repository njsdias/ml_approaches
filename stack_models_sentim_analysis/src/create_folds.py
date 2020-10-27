import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":

    # load dataset
    df = pd.read_csv("../input/labeledTrainData.tsv", sep="\t")

    # create new column  named as k-fold with value -1
    # we can fill with other number because it will be changed
    # during the k-folder cross validation process
    df.loc[:, "kfold"] = -1

    # shuffle dataframe to guarantee that we are not picking
    # only labels with value 1 or only with value 0
    # This a way to gaurantee the model is not will be biased
    # by the sequence of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # save the column with labels in our prediction variable
    y = df.sentiment.values

    # initialize stratified k-fold method
    skf = model_selection.StratifiedKFold(n_splits=5)

    # split data using stratified k-fold method
    # f: folder; t_ : train, v_; validation
    for f, (t_, v_) in enumerate(skf.split(X=df, y=y)):
        # fill kfold column in validation
        # position with current fold number
        df.loc[v_, "kfold"] = f

    # save in a csv the data separated in folds
    df.to_csv("../input/train_folds.csv", index=False)

    # activate the next line ig you want to
    # check if the you have the data split in correct folds
    # print(df.kfold.value_counts())











