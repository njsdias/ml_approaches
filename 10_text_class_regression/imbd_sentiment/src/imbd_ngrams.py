import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
if __name__ == "__main__":
    # read the training data
    df = pd.read_csv("../input/imdb.csv")

    # map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    # k-fold section
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # randomize the row of the data to guarantee it is not in
    # sequential: first only positive second only negative records
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch labels: take only the values
    # that we want to predict
    y = df.sentimental.values

    # initiate the kfold class
    n_folds = 5
    kf = model_selection.StratifiedKFold(n_splits=n_folds)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # we go over the folds created
    for fold_ in range(n_folds):
        # temporary dataframes for train and test
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)

        # initialize CounterVectorizer
        tfidf_vec = TfidfVectorizer(tokenizer=word_tokenize,
                                    token_pattern=None,
                                    ngram_range=(1, 3))

        # fit count_vec on training data reviews
        tfidf_vec.fit(train_df.review)

        # transform training and validation data reviews
        xtrain = tfidf_vec.transform(train_df.review)
        xtest = tfidf_vec.transform(test_df.review)

        # initialize naive bayes model
        model = linear_model.LogisticRegression()

        # fit the model
        model.fit(xtrain, train_df.sentimental)

        # predictions
        preds = model.predict(xtest)

        # calculate accuracy
        accuracy = metrics.accuracy_score(test_df.sentiment, preds)

        print("f:Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")
