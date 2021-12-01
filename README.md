# SeniorProject2021
UF Senior Project

Senior Project for the class CIS/CEN4917 CISE Design & Senior Project -Fall 2021

The purpose of the project was to train models on sawgrass roots to compare to the previously created models which were trained on a larger sample of peanut roots and with transfer learning were trained on a smaller sample of sawgrass roots.

A demo of the models can be ran from Demo.py or from Demo-jupyter.ipynb

Due to GitHub size limitations, the models are stored on a Google Drive. The models can be downloaded from here:
https://drive.google.com/file/d/17xJO8T_hJljmVOEtSTG4YHFgYjjZ-MJA/view?usp=sharing

Exmaple images of the demo running:

![image](https://user-images.githubusercontent.com/44200079/144148773-c795ecd1-2bfb-466c-b0d6-641728f4b9f6.png)
![image](https://user-images.githubusercontent.com/44200079/144148794-bdbb4546-5f5c-4d6c-98f2-02c2949a191d.png)


The fileCleaner will take PAT files and check to see if previous root data is in PAT files. 

The GenerateGT will take the PAT files and create the Ground Truth images if there is a data in the PAT file. 

For fileCleaner to work, the entire dataset should be given, since it realizes on comparing other PAT files.

PAT files to be cleaned should be placed into "unclean/pat" and the files will output into "pat"

GeneratorGT reads from "img" for the images and "pat" of the PAT files.

Both the pat file and the image must have the same name for the GT to be made. 


Before Cleaning:

![GT_BRUROOTSTUDY_T001_L005_2019 09 26_091748_006_HR](https://user-images.githubusercontent.com/44200079/144151647-d8a192cd-c457-466a-bb67-7becc604f977.png)

After Cleaning:

![BRUROOTSTUDY_T001_L005_2019 09 26_091748_006_HR](https://user-images.githubusercontent.com/44200079/144151666-402c27bd-b110-413e-9d5d-fff0a0b3f868.jpg)

Training data from testing the models:
https://wandb.ai/plundgren/U-Net?workspace=user-plundgren

The previous work that this project is based on: 
https://github.com/GatorSense/PlantRootSeg

Where the original files for the training and evaluation code had come from: https://github.com/milesial/Pytorch-UNet

