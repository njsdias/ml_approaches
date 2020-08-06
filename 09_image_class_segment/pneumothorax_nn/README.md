# Problem Description

The main propose of this ML algorithm is classify images.

The dataset is compounded by images of pneumorothorax. The dataset has images
where is identified problems in lung and it is divied in two categories: 
- have pneumothorax
- not have pneumothorax 

Dataset is available at: 

https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation  

To have png images you need to use the script that you can find in
https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data?select=download_images.py 

and that is in `src` folder of this project.



# Loss Function

Because our dataset set is a binary classification the **Loss Function** used was the Binary Cross Entropy (BCE) .
In this case was used `BCEWithLogitsLoss` that needs the logits as inputs instead of outputs of Sigmoid,
since it will apply sigmoid internally.  

# Code Structure
 
The first step is to
create a folds file, i.e. train.csv but with a new column kfold.


The dataset class  (`datatset.py`) will return an
item or sample of data. This sample of data should consist of everything you need
in order to train or evaluate your model.

The `engine.py` has training and evaluation functions. Let’s see
how engine.py looks like.

The `model.py` is where we develop our model. Develpo model in a separated file allow us
allows us to easily experiment with different models and different architectures 
(AlexNet, ResNet, DenseNet, etc). When we use a pre-trained model (ImageNet) we can use the weights of the model. However, when we not
use these weights, we allow our network learn everything from scratch.

After develop a model we can start training. 
In `training.py` we load the train.csv and all _.png_ files. After that we train the model and
do validation. The AUC ROC Curve metric allow us to see the model's performance. 




