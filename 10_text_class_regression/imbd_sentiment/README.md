# Problem Description

IMBD dataset contains the description of movies and a tag that classifies if
a movies has a positive or negative sentiment.

The ratio
to positive and negative samples in this dataset is 1:1, and thus, we can use accuracy
as the metric. We will use StratifiedKFold and create a single script to train five 
folds.

**Download Dataset**
- https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Models

Models used for this problem:

- Logist Regression: `imbd_lr.py`. Took a lot of time to train due the sparse matrices
produced by `CounterVectorized` libray.
- Naïve Bayes: `imbd_nb.py`. Naïve
bayes classifier is quite popular in NLP tasks as the sparse matrices are huge and
naïve bayes is a simple model.
- TF-IDF: `imbd_tfidf.py`. It is calculate using the follow information:
    - TF = (Number of times a term X appears in a document) / (Total numbers of terms in the document)
    - IDF = (Total number of documents) / (Number of documents with term X in it )
    - TF-IDF(X) = TF(X) * IDF(X)
- N-grams: it combines words in order. We can group how many words we want. 
But we need to pay attention to the order. Until now we were 
considered only one word (one-gram). With N-gram we can 
considered a sets of words that become a part of our vocabulary. 
We can use n-gram as a parameter in CountVectorizer and TfidfVectorize. By default 
the minimum and maximum limit are (1,1). We can change it to (1,3).

    
# Results

|Model|Accuracy| Speed
|-----|-----|-----|
|Logistic Regression| 0.8914| Slow |
|Naïve Bayes| 0.8455| Fast|
|TF-IDF| 0.8995| ---|
|N-grams| 0.8847| ---|
|Logistic Regression + FastText| 0.8595

# Stemming and Lemmatization
These techniques reduce the word to its smallest form.
- Lemmatization is more agressive than Stemmig.
- Stemming is more popular and widely used.
- Types of stemmers and lemmatizers:
    - Snowball Stemmer
    - WordNet Lemmatizer
    
# Topic Extraction
 It can be done with decompositoin techniques that reduces the data to
 a given number of components:
 - non-negative matrix factorization
 - latent semantic analysis (LSA) known as Singular Value Decomposition (SVD)
 
 See the file: `imbd_svd.py`
 
 
 # Removing Stopwords
 
 These words are identified as having high frequency in language.
 In English language them can be as: 
 - a, an, the, for
 
Removing this words can turn your sentence less understandable :
- “I need a new dog” -> “need new dog”

# Word Embeddings

It is the process to transform a word in a vector of numbers rather than transform a word in a single 
number between 0 to N-1, with N as unique tokens in a given corpus. We have some approaches available:
- Word2Vec: from Google
- FastText: from Facebook
- Glove (**Glo**bal **Ve**ctors for Word Representation): from Standford


# CBoW Model: Continuous Bag of Words

Using a simple network to learn the embeddings for words by reconstruction fo an input sentences.
The process consists in predict the missing word using all words. 

# Skip-gram Model

Using s simple network to capture the context of words taking one word.

# How word embeddings can learn?

The process to transform a word in numerical vector can learn using
- CBow and Skip-gram for Word2Vec
- FastText learns embeddings for character n-grams
- GloVe leas these embeddings by using co-occurence matrices 

At the end we obtain a dictionary where the key is a word in the corpus and
value is a vector of size N (300).

# Transform word vector into sentence word
After we have the numerical vectors for each words in a sentence we can 
create a normalized vector from all words vectors of the tokens (words).
In this way we obtain a **sentence vector**.

You can see this procedure in `ìmbd_sentence_vector.py`
You can download the embeddings for FastText from:
- https://fasttext.cc/docs/en/english-vectors.html

and save it in your `ìnput` folder.

# Text are like time series
Any sample is a sequence of tokens at different timestamps which are in increasing order,
an each token(word) can be represented as a vector (embeddings).

This way to interpret a text can allow us to apply models that are widely
used for time series:
- LSTM: **L**ong **T**erm **S**hort **M**emory
- GRU: **G**ated **R**ecurrent **U**nits
- CNN: **C**onvolutional **N**eural **N**etworks 





 
 


