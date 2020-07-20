The classification problem for MNIST is solved using
Decision Tree Classifier.

The datset file was download at

- https://www.kaggle.com/oddrationale/mnist-in-csv/data?select=mnist_train.csv

The columns are:
- label, pixel0, pixel1, pixel3, ...pixel783

In input folder are saved:
- raw data: mnist_train.cvs
- folds data generated using raw data: mnist_train_folds.csv

The models are saved in models folder

In source folder are saved:
- **config.py**: Models / Location constants
- **create_folds.py**: Function to create mnist_train_folds.csv file 
- **model_dispatcher.py**: imports `tree` and `ensemble` from scikit-learn and defines a dictionary with
keys that are names of the models and values are the models themselves. Here, we
define 
    - two different decision trees, one with gini criterion and one with entropy.
    - and one ensemble model: Random Forest
- **train.py**: train model and make predictions.


To run the program:
- In terminal inside of src folder execute the command:
 
 `pyhton3 train.py --fold NUM --model MODEL_CRITERION`
 
 in which `NUM` can assume values from `0` to `4` and
 `MODEL_CRITERIUM` assume next values:
 - `decision_tree_gini`
 - `decision_tree_entropy`
 - `rf`
 
These results were obtained using Decision Tree, with Gini criterion:
- Fold=0, Accuracy=0.8683
- Fold=1, Accuracy=0.8680
- Fold=2, Accuracy=0.8652
- Fold=3, Accuracy=0.8660
- Fold=4, Accuracy=0.8674 

These results were obtained using Random Forest:
- Fold=0, Accuracy=0.9704
- Fold=1, Accuracy=0.9665
- Fold=2, Accuracy=0.9679
- Fold=3, Accuracy=0.9663
- Fold=4, Accuracy=0.9663 