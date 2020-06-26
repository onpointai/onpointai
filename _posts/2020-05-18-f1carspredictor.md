---
toc: true
layout: post
description: Predict the constructor of the F1 car  from its image (2017-2020 season)
image: images/f1cardpred.png
categories: [AI/ML, Image classification]
title: Predict F1 Car Constructor from Image
---

## Overview  

This is an exercise based on Formula 1 cars and is  aimed at predicting which team a car belongs to by analysing an image. The image can be from any angle. This is an end-to-end example ir from gathering the data, training a model and deploying the trained model using a webapp on a local server.

## Setup  

The training setup, data are in this [repo](https://github.com/onpointai/f1cars-detector)

This is a good example of a complete workflow
  - collecting data using google web search (local) 
  - cleaning the data (removing duplicate images and other rubbish) (local) 
  - porting data to Google Colab - When i was doing this I did not have my Github a/c organised properly  
  Otherwise I would have done git init and ported the local repo to github  
  From Google Colab it is easy to clone the Github repo 
  - Once the data is in repo start a new jupyter notebook and select a GPU for training.  
  - Download the trained model to the local dir  
  - Modify the webapp to point to the trained model and do the predictions
  
Note: I only used about 75 images per class (10 classes) to cut down on the training time.

I used Fastai's recommended Starlette web api and with a bit of playing around with the css and html file got something suitable.

This is only meant for demo purposes and to motivate me to carry on with the ret of the course.

## Results  

![]({{"/"|relative_url}}/images/f1carpred-fp-1.png "Start")
![]({{"/"|relative_url}}/images/f1carpred-fp-3.png "Output")

---

 > Suggested Improvements:
 - [ ] Add more training data
 - [ ] Clean training data 
 - [ ] Overfit and then play around with hyperparameters
 - [ ] Give webapp better UI
 
---

![]({{"/"|relative_url}}/images/onpointai_logo.gif)
