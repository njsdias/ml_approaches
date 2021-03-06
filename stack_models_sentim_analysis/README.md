# Ensembling, Blending & Stacking Models

Here is showed how we can combine several models to do predictions.

The example shows it is not need to use fancy models (as BERT) to do sentimental
analysis.

The dataset used is the well known IMDB that contains movies reviews labeled as 
positive/negative sentiment reviews.

To overcome the imbalanced dataset, it was used the k-fold cross validation method
to split that in test and train.

The dataset can be download [here](https://www.kaggle.com/c/word2vec-nlp-tutorial/data).

# Execution Sequence

- create_folds.py: split data using k-folder cross validation 
- lr.py: use TF-IDF to vectorized the text reviews and use Logistic regression model to predict the labels
- lr_cnt.py: use Count Vectorized to vectorized the text reviews and use Logistic regression model to predict the labels
- rf_svd.py: Random Forest with Singular Vector Decomposition used to reduce feature space

This files are related to blending the different results
- blending.py: Evaluate the AUC average of these three models
- optimal_weighted.py: optimize the model coefficients to compare with the blending.py results 
- lr_blend.py


# Files Generated
Inside of models_preds folder you can find the files
related with the results generated by different models.
- lr.csv 
- lr_cnt.csv 
- rf_svd.csv 

The models are saved in the follow files:
- lr_model.sav
- lr_cnt_model.sav
- rf_svd_model_sav   
 
To use the models:

    # load the model from disk
    loaded_model = joblib.load(filename)
    result = loaded_model.score(X_test, Y_test)
    print(result)


# See some model results
It is possible to explore the results of the model
only to have a raw idea of the results.

Inside of the models_pred folder you can see the head of
the predictions:

    head ../models/pred/lr_cnt.csv 
    
