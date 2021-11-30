#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:57:06 2018

@author: weihuangxu
@modified by: Paul Lundgren
"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pdb
import cv2
blank = Image.open("blank.jpg")
# find the x range and y range of two given points
def linearFunction(P1, P2):
    
    jud = P2[0]-P1[0] #x
    if jud == 0:
        yRange = np.arange(min(P1[1], P2[1]),max(P1[1], P2[1])+1)
        xRange = P2[0]*np.ones((len(yRange)))
    else:
        k = (P2[1]-P1[1])/jud #slope
        xRange = np.arange(min(P1[0], P2[0]), max(P1[0], P2[0])+1)
        yRange = np.round(k*(xRange-P1[0]))+P1[1]
    
    xyRange = np.stack((xRange, yRange), axis=-1).astype(int)
    return xyRange

# find the area indexes for rectangle shape
def Rectangle(vertex):
    
    vertex = vertex.astype(int)
    A = vertex[0:2] 
    C = vertex[2:4] 
    B = vertex[4:6] 
    D = vertex[6:8] 
    
    ABRange = linearFunction(A,B)
    CDRange = linearFunction(C,D)
    CBRange = linearFunction(C,B)
    ADRange = linearFunction(A,D)
    
    allRange = np.vstack((ABRange, CDRange, CBRange, ADRange))
    xmin = min(allRange[:,0])
    xmax = max(allRange[:,0])
    
    allx = []
    ally = []
    
    for i in range(int(xmin), int(xmax+1)):

        yRange = allRange[np.where(allRange[:,0] == i), 1].squeeze()
        ymin = min(yRange)
        ymax = max(yRange)
        y = np.arange(ymin, ymax+1)
        x = i * np.ones(np.shape(y)).astype(int)
        allx += x.tolist()
        ally += y.tolist()
    
    #area = np.hstack(allx, ally)
    
    return allx, ally

def del_empty(path, print_result=False):
    """
    Delete the empty label files

    Parameters
    ----------
    path : str, the directory of all the data
    print_result : Boolean. print the progress or not

    """
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    
    count = 0
    totLen = len(listOfFiles)
    # import pdb; pdb.set_trace()
    for item in listOfFiles:
        label = np.loadtxt(item, dtype='str')
        Rlocation = np.core.defchararray.find(label, 'R')
        # find the idx of string starting with R
        Ridx = np.where(Rlocation == 0)
        if len(Ridx[0]) <= 0:
            os.remove(item)
            count += 1
            if print_result:
                print('%d/%d deleted: %s.' %(count, totLen, item.split('/')[-1]))
            

def generate_mask(label, img_size, black_root=True):
    '''
    Generate binary mask for Minirhizotron images using .pat label file.

    Parameters
    ----------
    label : np.array. array of string 

    img_size : tuple, the image size.

    Returns: the binary mask of root of 1 and background of 0
    '''
    # pdb.set_trace()
    h, w, c = img_size
    GT = np.zeros((h,w), dtype=(np.uint8))
    # find the string start with 'R', -1 is the string not start with R and 0 is the string start with R
    Rlocation = np.core.defchararray.find(label, 'R')
    # find the idx of string starting with R
    Ridx = np.where(Rlocation == 0)
    # if len(Ridx[0]) <= 0:
    #     return
    
    numRec = np.shape(Ridx)[1]
    vertex = np.zeros((numRec, 8))
    
    xidx = []
    yidx = []
    for i in range(numRec):
        idx = Ridx[0][i] #index of 'R1,2..."
        vertex[i,:] = label[idx+9:idx+17]
        tempx, tempy = Rectangle(vertex[i])
        xidx += tempx
        yidx += tempy
    
    yidx = np.asarray(yidx) 
    xidx = np.asarray(xidx)
    yidx[np.where(yidx >= h)] = h - 1
    xidx[np.where(xidx >= w)] = w - 1
    allindex = (yidx, xidx)
    
    GT[allindex] = 255
    if black_root:
        GT = 255 - GT
    return GT

# import pdb; pdb.set_trace()

image_dir = 'C:\\Users\\Paul\\PatCleaning\\img'
label_dir = 'C:\\Users\\Paul\\PatCleaning\\pat'
out_dir = 'C:\\Users\\Paul\\PatCleaning\\complete'
label_lt = list()
count = 0
remove_label = []
empty_label = []
for (dirpath, dirnames, filenames) in os.walk(label_dir):
    label_lt += [os.path.join(dirpath, file) for file in filenames]

for item in label_lt:
    if item.endswith('.pat'):
        img_name = item.split('\\')[-1]
        img_dir = os.path.join(image_dir, img_name.replace('.pat', '.jpg'))
        if not os.path.exists(img_dir):
            remove_label.append(item)
            continue
        img_name = img_dir.split('\\')[-1] # get the image name
        label = np.loadtxt(item, dtype='str')
        Rlocation = np.core.defchararray.find(label, 'R')
        Ridx = np.where(Rlocation == 0)
        if len(Ridx[0]) <= 0:
            empty_label.append(item)
            img_out_dir = os.path.join(out_dir, 'no_root')
            #GT_name = img_name.replace('.jpg', '.bmp') # get the new GT name and extension
            #GT_out_dir = os.path.join(out_dir, 'GT', GT_name)
            #GT = blank
           # GT.save(GT_out_dir)
            if not os.path.exists(img_out_dir):
                os.makedirs(img_out_dir)
            shutil.copy(img_dir, img_out_dir)
            continue;
        else:
            # pdb.set_trace()
            print('Processing %s' %(item))
            rawImg = mpimg.imread(img_dir) # load the raw image
            GT_name = img_name.replace('.jpg', '.jpg') # get the new GT name and extension
            GT_out_dir = os.path.join(out_dir, 'GT', GT_name)
            GT = generate_mask(label, rawImg.shape)
            if not os.path.exists(GT_out_dir[:-len(GT_name)]):
                 os.makedirs(GT_out_dir[:-len(GT_name)])
            # plt.imsave(GT_out_dir, GT, cmap='gray')
            cv2.imwrite(GT_out_dir, GT)
            
            # save out the raw image to desired folder
            img_out_dir = os.path.join(out_dir, 'has_root', img_name)
            if not os.path.exists(img_out_dir[:-len(img_name)]):
                 os.makedirs(img_out_dir[:-len(img_name)])
            shutil.copy(img_dir, img_out_dir)
        
        count += 1
        print('%d/%d is finished.'%(count, len(label_lt)))
            
            
            

        