import pandas as pd
from ntkl.tokenize import word_tokenize
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
import regex as re
import string

# If you run the below code without a clean text procedure
# you will see the results don't make sense because
# it mixed punctuation with words in a not clean way.
# Thus we can clean the text first.
# In general, we start by cleaning text after we
# do text analyses.


def clean_text(s):
    """
    This function cleans the text a bit
    :param s: string
    :return: cleaned string
    """
    # remove all punctuations using regex and string module
    s = re.sub(f'[{re.escape(string.punctuation)}]', '', s)

    return s


# load dataset with only 10k rows
corpus = pd.read_csv("../input/imbd.csv", nrows=10000)

# clean the text
corpus.loc[:, "review"] = corpus.review.apply(clean_text)

# collect only the text in review column
corpus = corpus.review.values

# initialize TfidfVectorizer
tfv = TfidfVectorizer(tokenizer=word_tokenize,
                      token_pattern=None)

# fit the vectorizer to corpus
tfv.fit(corpus)

# transform the corpus using tfidf
corpus_transformed = tfv.transform(corpus)

# initialize SVD with 10 components
svd = decomposition.TruncatedSVD(n_components=10)

# fit SVD
corpus_svd = svd.fit(corpus_transformed)

# choose first sample and create a dictionary
# of feature names and their scores from svd
# you can change the sample_index variable to
# get dictionary for any other sample
sample_index = 0
feature_scores = dict(zip(tfv.get_feature_names(),
                         corpus_svd.components_[sample_index]))

# once we have the dictionary, we can now
# sort it in decreasing order and get the
# top N topics
N = 5
print(sorted(feature_scores, key=feature_scores.get, reverse=True)[:N])

# run it for multiple samples
N = 5
for sample_index in range(N):
    feature_scores = dict(zip(tfv.get_feature_names(),
                         corpus_svd.components_[sample_index]))
    print(sorted(feature_scores, key=feature_scores.get, reverse=True)[:N])





