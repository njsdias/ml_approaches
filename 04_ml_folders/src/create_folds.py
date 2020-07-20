import pandas as pd
from sklearn import model_selection


# Training file was downloaded at
# https://www.kaggle.com/oddrationale/mnist-in-csv/data?select=mnist_train.csv
# header: [label, pixel0, pixel1, pixel2, ..., pixel783]

def create_folds():

    # Read mnist_train_csv
    df = pd.read_csv("../input/mnist_train.csv")

    # create column called kfold
    df["kfold"] = -1

    # shuffle/randomize rows
    df = df.sample(frac=1).reset_index(drop=True)

    # initiate the kfold class from model_selection module
    kf = model_selection.KFold(n_splits=5)

    # Create file in /input folder called mnist_train_folds.csv
    for fold, (trn_,val_) in enumerate(kf.split(X=df)):
        df.loc[val_,'kfold'] = fold

    # save the new file with kfold column
    return df.to_csv("../input/mnist_train_folds.csv", index=False)
