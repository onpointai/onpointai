---
toc: true
layout: post
description: Predict the sports activity in a short video clip
image: images/videoclassif.png
categories: [AI/ML, Video Classification]
title: Sports Video Classification
---
## Overview      
The simplest way to classify the type of video is by examining individual frames of the video ie treat individual frames as images, classify the images and then do some sort of averaging or smoothing over a few frames to predict the current display.

Once we understand what goes on here  we can explore the [other](https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5) methods which take up more setup and compute.

The main package here is [OpenCV](https://opencv.org), a library aimed at real-time computer vision tasks.

The dataset used was a collection of images curated using Google image search.

├── Sports-Type-Classifier  
│   ├── data  
│   │   ├── badminton [938 entries]  
│   │   ├── baseball [746 entries]  
│   │   ├── basketball [495 entries]  
│   │   ├── boxing [705 entries]  
│   │   ├── chess [481 entries]  
│   │   ├── cricket [715 entries]  
│   │   ├── fencing [635 entries]  
│   │   ├── football [799 entries]  
│   │   ├── formula1 [687 entries]  
│   │   ├── gymnastics [719 entries]  
│   │   ├── hockey [572 entries]  
│   │   ├── ice_hockey [715 entries]  
│   │   ├── kabaddi [454 entries]  
│   │   ├── motogp [679 entries]  
│   │   ├── shooting [536 entries]  
│   │   ├── swimming [689 entries]  
│   │   ├── table_tennis [713 entries]  
│   │   ├── tennis [718 entries]  
│   │   ├── volleyball [713 entries]  
│   │   ├── weight_lifting [577 entries]   
│   │   ├── wrestling [611 entries]  
│   │   ├── wwe [671 entries]    

## Setup   
The [repo](https://github.com/onpointai/keras-video-classification) contains the data, Colab Jupyter notebooks and trained weights for different training regimes.

## Results  

[![Input](http://img.youtube.com/vi/TZhSxFWh7wc/0.jpg)](http://www.youtube.com/watch?v=TZhSxFWh7wc "Input")  
[![Prediction](http://img.youtube.com/vi/i9O1bb89Z9c/0.jpg)](http://www.youtube.com/watch?v=i9O1bb89Z9c "Prediction")
              
---
[![Input](http://img.youtube.com/vi/HjaCPFLzzLI/0.jpg)](http://www.youtube.com/watch?v=HjaCPFLzzLI "Input")  
[![Prediction](http://img.youtube.com/vi/U03uXXqwbA8/0.jpg)](http://www.youtube.com/watch?v=U03uXXqwbA8 "Prediction")
        
---
[![Input](http://img.youtube.com/vi/QaEUxM0N0p8/0.jpg)](http://www.youtube.com/watch?v=QaEUxM0N0p8 "Input")  
[![Prediction](http://img.youtube.com/vi/G6YeY-KNuJM/0.jpg)](http://www.youtube.com/watch?v=G6YeY-KNuJM "Prediction")

---
![]({{"/"|relative_url}}/images/onpointai_logo.gif)
