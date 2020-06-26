---
toc: true
layout: post
description: Using Pytorch to develop a Deep Learning framework to predict pneumonia in X-ray images
image: images/xray-pneumonia-1.png
categories: [jovian, pytorch, transfer learning, fastpages, jupyter]
title: Using Deep Learning to detect Pneumonia in X-ray images
sticky_rank: 2
---

# About

This blog is towards the [Course Project](https://jovian.ml/forum/t/assignment-5-course-project/1563) for the [Pytorch Zero to GANS] free online course(https://jovian.ml/forum/c/pytorch-zero-to-gans/18) run by [JOVIAN.ML](https://www.jovian.ml).

The course [competition](https://jovian.ml/forum/t/assignment-4-in-class-data-science-competition/1564/2) was based on analysing protein cells with muti-label classification.

Therefore, to extend my understanding of dealing with medical imaging I decided to use the [X-Ray image database](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) in Kaggle.

Seeing as I ran out of GPU hours on Kaggle because of the competition (restricted to 30hrs/week at the time of writing June 2020) I opted to use Google Colab. 

This blog is in the form of a Jupyter notebook and inspired by [link](https://github.com/viritaromero/Detecting-Pneumonia-in-Chest-X-Rays/blob/master/Detecting_Pneumonia.ipynb).

The blog talks about getting the dataset in Google Colab, explore the dataset, develop the training model, metrics and then does some preliminary training to get a model which is then used to make a few predictions. 
I will then talk about some of the lessons learned.

> Warning! The purpose of this blog is to outline the steps taken in a typical Machine Learning project and should be treated as such.

**Link to non-sanitised notebook on Jovian.ML here **

# Import libraries
```

```
# Colab setup and getting data from Kaggle

I used Google Colab with GPU processing for this project because I had exhausted my Kaggle hours (30hrs/wk) working on the competition :( The challenge here was signing into Colab, setting up the working directoty and then linking to Kaggle and copying the data over. The size of the dataset was about 1.3Gb which wasn't too much of a bother as Google gives each Gmail account 15Gb for free!


> Tip: I used the monokai settings in Colab which gave excellent font contrast and colours for editing.
![monokai]({{"/"|relative_url}}/images/xray-colab-monokai.png "Colab Monokai Setting")

---

Hope you enjoyed it

---

![]({{"/"|relative_url}}/images/onpointai_logo.gif)

