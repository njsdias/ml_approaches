# Problem Description

In this folder the IMBD classification problem is solved using Transformers.
Transformers can look at all the words in the whole sentence simultaneously. 

Please, for more details read the book. The book give us more insights.
 

# Files Description
The files description is presented following the implementation sequential, as
reported in the book. 

- `config.py`: set up the main configuration for the model and train
- `dataset.py`: this class class return all what we need to train the model
- `model.py`: in this class we build the model
- `engine.py`: The previous model returns a single output. Thus, we can use binary cross-entropy loss with
logits which first applies sigmoid and then calculates the loss.
- `train.py`: in this file we build our train model strategy.

The accuracy of this model is 93%, which is better than the precious model that
we saw in this chapter.

One note from the author is that LSTM model gaves us 90% of accuracy. This model is much simple simpler, easier
to train and faster when it comes to inferene. We can improve the model (read book
to know how). The best advice is: **Don't choose BERT only because it's "cool"**.




 