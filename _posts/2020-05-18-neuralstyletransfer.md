---
toc: true
layout: post
description: Desc - Render F1 car images in the style of Picasso
image: images/styletransfer.png
categories: [AI/ML, Style Transfer]
title: Style Transfer using Neural Networks
---

## Overview  

This method renders the contents of a Original image in the style of a Reference image. For example in an image of a cityscape the contents can be considered as the hard edges of the buildings. The Reference image can contain lots of swirly patters and colours which will be know as the style of the image. 
The loss function is defined in parameter space as the difference between the content and style of the images as laid out below:    
```
Loss = distance((style(reference_image) - style(generated_image)) +  
        (distance(content(original_image) - content(generated_image))

```

As a workflow exercise I attempted to render the image of an F1 car in the style of Picasso!
I used images of recent Racing Point and Renault F1 cars.

## Setup  
The notebook, models and data are contained [here](https://github.com/onpointai/f1car-style-transfer)

I trained it on Google Colab using both Tensorflow and Pytorch (separately ofcourse!)
The trained model is implemented using a webapp. The user is asked to select an image and the analysis is displayed.

---

## Results    

![]({{site.baseurl}}/images/rp-style-transfer.png "Racing Point")
![]({{site.baseurl}}/images/renault-style-transfer.png "Renault")


```
Loss = distance((style(reference_image) - style(generated_image)) +  
        (distance(content(original_image) - content(generated_image))
        
original_image = a picture of an F1 car 
reference_image = a style image such as Picasso

```
---
![]({{"/"|relative_url}}/images/onpointai_logo.gif)
