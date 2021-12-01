# reading files
"""
@author:  Paul Lundgren
"""
import os
import numpy as np
import copy
base_dir = 'pat'  #output location
label_dir = 'unclean' #input location
label_lt = list()
list_lt = list()
cleaned_list_lt = list()
filename_lt =list()
count = 0
remove_label = []
empty_label = []
area_list = list()
print("Loading PAT files")
for (dirpath, dirnames, filenames) in os.walk(label_dir):
    filename_lt = filenames
    label_lt += [os.path.join(dirpath, file) for file in filenames]
for item in label_lt:
    label = np.loadtxt(item, dtype='str')
    Rlocation = np.core.defchararray.find(label, 'R')
    Ridx = np.where(Rlocation == 0)
    numRec = np.shape(Ridx)[1]
    vertex = np.zeros((numRec, 8))
    area = list()
    xidx = []
    yidx = []
    for i in range(numRec):
        idx = Ridx[0][i] #index of 'R1,2..."
        vertex[i,:] = label[idx+9:idx+17]
        area.append((.5) * (((vertex[i,0]*vertex[i,3]) - ((vertex[i,2]*vertex[i,1]))) + ((vertex[i,2]*vertex[i,5]) - ((vertex[i,4]*vertex[i,3]))) + ((vertex[i,4]*vertex[i,7]) - ((vertex[i,6]*vertex[i,4]))) + ((vertex[i,6]*vertex[i,1]) - ((vertex[i,7]*vertex[i,0])))))
    area_list.append(area)
    list_lt.append(vertex)
blanklist = [0, 0, 0, 0, 0, 0, 0, 0]
cleaned_list_lt = copy.deepcopy(list_lt)
cleaned_area_list = copy.deepcopy(area_list)
print("Cleaning PAT files")
for i in range(1,len(area_list)):
    tempone = list_lt[i-1]
    temptwo = cleaned_list_lt[i]
    areaone = area_list[i-1]
    areatwo = cleaned_area_list[i]
    for x in range(len(areaone)):
        for y in range(len(areatwo)):
            if(areaone[x] == areatwo[y]):
                temptwo[y] = blanklist
    cleaned_list_lt[i] = temptwo
print("making new PAT files")
for item in label_lt:
    label = np.loadtxt(item, dtype='str')
    Rlocation = np.core.defchararray.find(label, 'R')
    Ridx = np.where(Rlocation == 0)
    numRec = np.shape(Ridx)[1]
    vertex = np.zeros((numRec, 8))

    xidx = []
    yidx = []
    temp = cleaned_list_lt[count]
    for i in range(numRec):
        idx = Ridx[0][i] #index of 'R1,2..."
        label[idx+9:idx+17] = temp[i,:]
    fullpath = os.path.join(base_dir,filename_lt[count])
    textfile = open(fullpath, "w")
    for element in label:
        textfile.write(element + "\n")
    textfile.close()
    count = count + 1
