# Problem Description

Captcha gives a images with letters and numbers and ask you to write what you are see in the images
to identify if you are a human or not.

In this dataset the name of the files is the label of each example. 

This is a classification problem that we need to predict different characters. 


**Download data**

Inside of your `input` folder run the follow command to download dataset:

`wget https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip`

When we unzip the file it will automatically creates a folder named as `captcha_images_v2`. To unzip file run the follow
command inside of `ìnput` folder:

`unzip captcha_images_v2.zip `

The objective is predict the characters that are in each captcha (image).

  
# Data Description
The data are captcha images with different characters.
We have 19 unique characters.
The name of each file, is the captcha that is in the image. 



# Loss Function

Here was used CTC loss that is used for sequences, because we have  to predict
sequence of characters.

CTC (Connectionist Temporal Classification) used to train deep neural networks in speech
recognition, handwriting, recognition and other sequence algorithms.



# Results
We need to investigate a new ways of extract data from predictions using different
 CTC libraries decoders for have a better predictions.

# Additional Information
For more information: https://www.youtube.com/watch?v=IcLEJB2pY2Y&list=PLjMBCjnfVRHQZGxbCcpd41Fm4nfBPVnCa&index=38&t=6s