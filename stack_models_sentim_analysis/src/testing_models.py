import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

if __name__ == "__main__":

    # load dataset
    df = pd.read_csv("../input/labeledTrainData.tsv", sep="\t")

    # load new review
    filename = "review_terminator_bad.txt"
    review = open("../reviews/" + filename , "r")

    movie_name = filename.split(sep="_")[1]

    # vectorized texts
    # tfv = TfidfVectorizer(max_features=1000)
    # tfv.fit(df.review.values)

    # save tfidf model
    #filename = "../model_preds/tfidf.sav"
    #joblib.dump(tfv, filename)

    # load tdidf model
    tfv = joblib.load("../model_preds/tfidf.sav")
    review_vect = tfv.transform(review)

    # load model
    model = joblib.load("../model_preds/lr_model.sav")
    result = model.predict_proba(review_vect)[:, 1]

    print(result)

    if result > 0.8:
        print(f"{movie_name}: positive")
    else:
        print(f"{movie_name}:negative")



