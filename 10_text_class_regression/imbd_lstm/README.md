# Problem Description

IMBD dataset contains the description of movies and a tag that classifies if
a movies has a positive or negative sentiment.

The ratio
to positive and negative samples in this dataset is 1:1, and thus, we can use accuracy
as the metric. We will use StratifiedKFold and create a single script to train five 
folds.

**Download Dataset**
- https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Text are like time series
Any sample is a sequence of tokens at different timestamps which are in increasing order,
an each token(word) can be represented as a vector (embeddings).

This way to interpret a text can allow us to apply models that are widely
used for time series:
- LSTM: **L**ong **T**erm **S**hort **M**emory
- GRU: **G**ated **R**ecurrent **U**nits
- CNN: **C**onvolutional **N**eural **N**etworks 

# Problem Approach

This demo shows how we can use LSTM to predict the sentiment in a movie description.
or a negative sentiment.

# File Description

- `create_folds.py`: first step will be splitting the data for cross-validation
- `dataset.py`: creates a dataset class that returns one sample of the training 
or validation data
- `lstm.py`: LSTM model 
- `engine.py`: which consists of our training and evaluation functions
- `train.py`: it is used for training multiple folds
    - my advice is start building the train loop for each epoch add what you need
    above from it. 
- `config.py`: it is here where we define the configuration

**End note**

Please, read the book and try to understand in some deeper detail what is happening
in the code, and how each .py file is linked.




 
