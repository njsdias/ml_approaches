The objective of this project is to predict the target
variable with a data set that is compounded only by categorical features.

The dataset is available in:
https://www.kaggle.com/c/cat-in-the-dat-ii

## Dataset Description

The dataset consists of all kinds of categorical variables:
- Nominal: the order is not important (male, female)
- Ordinal: the order is important (novice, expert, contribute)
- Cyclical: days, months
- Binary

Overall, there are:
- Five binary variables
- Ten nominal variables
- Six ordinal variables
- Two cyclic variables
- And a target variable

For categories that are in text we need to convert them into numbers.

By EDA we saw the target variables is skewed (**imbalanced**) and it is
a binary variable. Thus, this problem is a **binary classification**
and the best metric for this problems is **AUC metric**. 
For cross-validation is used `StratifiedKFold`. 

The Nan values in categorical features will be treated as a unknown category. 

## Run code
The -W ignore is to ignore all warnings

`python3 -W ignore model_file.py` in which, 
`model_file.py` can take the values:
- `ohe_logres.py` : Logistic Regression with One-Hot Encoder
- `lbl_rf.py`: Random Forest with Label Encoder
- `ohe_svd_rf.py`:  Random Forest with One-Hot Encoder and Singular Value 
- `lbl_xgb.py`: XGBoost
- `entity_embeddings.py`: Neural Networks 


## Results

|Model|AUC|
|-----|-----|
|Logistic Regression| 0.7863|
|Random Forest with Label Encoder| 0.7156|
|Random Forest with SVD| 0.7064|
|XGBoost| 0.7650|


### Logistic Regression
Here, was used one-hot encoding to transform categorical features in string values.

The AUC values are similar in each fold. The average AUC is 0.7863 .

### Random Forest with Label Encoder
For applying random forest in this dataset, instead of one-hot encoding a `LabelEncoder`
was applied in every feature column to convert them in numeric values.

The AUC values are similar in each fold. The average AUC is 0.7156 .

By this results we can conclude:
- The Random Forest model, without any tuning of
hyperparameters, performs a lot worse than simple Logistic Regression.
- The compute time-consumption is higher compared with Logistic Regression
- A simple model like logistic regression performs a better. This is the reason because we cannot
simple models and we need start with them.

### Random Forest with SVD
The one-hot encode was applied to full data and then fit TruncatedSVD from scikit-learn on
sparse matrix with training + validation data. In this way, the high
dimensional sparse matrix was reduced to 120 features and then fit random forest classifier.

The AUC values are similar in each fold. The average AUC is 0.7064 . The results are worst.

### XGBoost

The AUC values are similar in each fold. The average AUC is 0.7650 .

### Neural Network with Entity Embedding

Here we represent categories by vectors with float values to reduce the matrix dimension as well 
computer time consumption.

Every category in a column is mapped to an embedding. After that the embeddings will 
be reshape to their dimension to make them flat. After that concatenate all flattened inputs
embeddings. The final step is adding a bunch of dense layers, an output layer.


