# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 01:09:16 2020

@author: yash1
"""

#python3 /u/yashkuma/cv_labs/lab1_filter.py /u/yashkuma/cv_labs

#Import the Image and ImageFilter classes from PIL ( Pillow ) 
from PIL import Image 
from PIL import ImageFilter
import random
import os
import numpy as np
import pandas as pd
from numpy import convolve
import sys

data_path = sys.argv[1]


# Lab 1 

# Convolve filters
# Load an image 
# Convert image into numpy array
im = Image.open(os.path.join(data_path, "bird.jpg"))
im.show() 
im_array = np.array(im)

# Defining kernels
id1 = np.array([[0,0,0],
              [0,1,0],
              [0,0,0]])
b = 1/9*np.array([[1,1,1],
                  [1,1,1],
                  [1,1,1]])
d = np.array([[0,0,0],
              [-1,0,1],
              [0,0,0]])

g = np.array([[0.003,0.013,0.022,0.013,0.003],
              [0.013,0.059,0.097,0.059,0.013],
              [0.022,0.097,0.159,0.097,0.022],
              [0.013,0.059,0.097,0.059,0.013],
              [0.003,0.013,0.022,0.013,0.003]]) 


def seperate_1d(kernel):
    U, D, V = np.linalg.svd(kernel)
    f1 = U[:,0] * np.sqrt(D[0])
    f2 = V[0] * np.sqrt(D[0])
    return f1, f2


def convolve_color_image(image, kernel):
    f1, f2 = seperate_1d(kernel)
    im_filter_array = np.zeros(image.shape)
    for i in range(len(image[0][0])):
        for k in range(len(image[0])):   
            im_filter_array[:,k,i] = np.convolve(im_array[:,k,i],f1,'same')
    for i in range(len(image[0][0])):
        for j in range(len(image)):   
            im_filter_array[j,:,i] = np.convolve(im_filter_array[j,:,i],f2,'same')
    im_filter_array1 = im_filter_array.astype('uint8')
    return im_filter_array1


def convolve_filters(k1,k2):
    a1, a2 = seperate_1d(k2)
    d_k1 = np.zeros(k1.shape)
    for k in range(len(g[0])):   
        d_k1[:,k] = np.convolve(k1[:,k],a1,'same')
    for j in range(len(g)):
        d_k1[j,:] = np.convolve(d_k1[j,:],a2,'same')
    return d_k1
        

# (a)
# Identity kernel

im_identity = convolve_color_image(im_array,id1)        
image_identity = Image.fromarray(im_identity)
image_identity.save(os.path.join(data_path, "bird_id.jpg"))
image_identity.show()  

# (b)
# Box blur

im_box = convolve_color_image(im_array,b)        
image_box = Image.fromarray(im_box)
image_box.save(os.path.join(data_path, "bird_box.jpg"))
image_box.show()  

# (c)
# Horizontal derivative

im_der = convolve_color_image(im_array,d)        
image_der = Image.fromarray(im_der)
image_der.save(os.path.join(data_path, "bird_der.jpg"))
image_der.show()  

# (d)
# Gaussian 

im_gaus = convolve_color_image(im_array,g)        
image_gaus = Image.fromarray(im_gaus)
image_gaus.save(os.path.join(data_path, "bird_gaus.jpg"))
image_gaus.show()  


# (e)
# Sharpening filter
alpha = 0.2
id2 = np.zeros(g.shape)
id2[2,2] = 1

s = ((1+alpha)*id2 - g)

im_sharp = convolve_color_image(im_array,s)        
image_sharp = Image.fromarray(im_sharp)
image_sharp.save(os.path.join(data_path, "bird_sharp.jpg"))
image_sharp.show()


# (f)
# Gaussian derivative

g_d = convolve_filters(g,d)

im_gausd = convolve_color_image(im_array,g_d)        
image_gausd = Image.fromarray(im_gausd)
image_gausd.save(os.path.join(data_path, "bird_gausd.jpg"))
image_gausd.show()  




  
     