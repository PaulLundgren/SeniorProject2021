# Import dependencies
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from UNetmodel import UNet as old_UNet
from unet import UNet as new_UNet
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def predict_img(net,
                full_img,
                device,
                classes,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), classes).permute(2, 0, 1).numpy()

Peanut_UnetD5 = './Models/Peanut_UnetD5.pth'
Sawgrass_UnetD4 = './Models/Sawgrass_UnetD4.pth'
Sawgrass_UnetD5 = './Models/Sawgrass_Unet_D5.pth'
Sawgrass_all_D4_e1 = './Models/Sawgrass_all_D4_e1.pth'
Sawgrass_all_D4 = './Models/Sawgrass_all_D4.pth'

device = torch.device('cpu')
UNetD5 = old_UNet(num_classes=1, depth=5)               #previous Unet model,
UNetD4 = new_UNet(n_channels=3, n_classes=2)            #new Unet model
UNetD5_test = old_UNet(num_classes=2, depth=5)            #new Unet model
image_path = '.\\Img\\testroots\\Image'
GT_path = '.\\Img\\testroots\\GT'
T = .5
for img in os.listdir(image_path):
    img_dir = os.path.join(image_path, img)
    GT_dir = os.path.join(GT_path, (os.path.splitext(img)[0] + '.jpg'))
    GT = Image.open(GT_dir).convert('L')
    original = Image.open(img_dir).convert('RGB')
    #Peanut_UnetD5_pred = predict_img(UNetD5,original,device,1)
    #Peanut_UnetD5_pred = mask_to_image(Peanut_UnetD5_pred)
    UNetD4.load_state_dict(torch.load(Sawgrass_UnetD4, map_location=device))
    Sawgrass_UnetD4_pred = predict_img(UNetD4,original,device,2)
    Sawgrass_UnetD4_pred = mask_to_image(Sawgrass_UnetD4_pred)
    UNetD5_test.load_state_dict(torch.load(Sawgrass_UnetD5, map_location=device))
    Sawgrass_UnetD5_pred = predict_img(UNetD5_test,original,device,2)
    Sawgrass_UnetD5_pred = mask_to_image(Sawgrass_UnetD5_pred)
    UNetD4.load_state_dict(torch.load(Sawgrass_all_D4_e1, map_location=device))
    Sawgrass_all_D4_e1_pred = predict_img(UNetD4,original,device,2)
    Sawgrass_all_D4_e1_pred = mask_to_image(Sawgrass_all_D4_e1_pred)
    UNetD4.load_state_dict(torch.load(Sawgrass_all_D4, map_location=device))
    Sawgrass_all_D4_pred = predict_img(UNetD4,original,device,2)
    Sawgrass_all_D4_pred = mask_to_image(Sawgrass_all_D4_pred)

    with torch.no_grad():
        input = ToTensor()(original).unsqueeze(0)
        UNetD5.load_state_dict(torch.load(Peanut_UnetD5, map_location=device))
        UNetD5.eval()
        output = UNetD5(input)
        Peanut_UnetD5_pred = output[0,0]
        Peanut_UnetD5_pred = torch.sigmoid(Peanut_UnetD5_pred)
        Peanut_UnetD5_pred = (Peanut_UnetD5_pred-Peanut_UnetD5_pred.min())/(Peanut_UnetD5_pred.max() - Peanut_UnetD5_pred.min())
        Peanut_UnetD5_pred = Peanut_UnetD5_pred < T
        input = ToTensor()(original).unsqueeze(0)
        plt.figure(frameon=False, figsize=(18,16))
        plt.subplot(3,3,1)
        plt.imshow(original)
        plt.axis('off')
        plt.title('Raw Image')
        plt.subplot(3,3,2)
        plt.imshow(GT, cmap='binary')
        plt.axis('off')
        plt.title('Ground Truth')
        plt.subplot(3,3,3)
        plt.imshow(Peanut_UnetD5_pred, cmap='binary')
        plt.axis('off')
        plt.title('Peanut_UnetD5_pred Result')
        plt.subplot(3,3,4)
        plt.imshow(Sawgrass_UnetD4_pred, cmap='binary')
        plt.axis('off')
        plt.title('Sawgrass_UnetD4_pred Result')
        plt.subplot(3,3,5)
        plt.imshow(Sawgrass_UnetD5_pred, cmap='binary')
        plt.axis('off')
        plt.title('Sawgrass_UnetD5_pred Result')
        plt.subplot(3,3,6)
        plt.imshow(Sawgrass_all_D4_e1_pred, cmap='binary')
        plt.axis('off')
        plt.title('Sawgrass_all_D4_e1_pred Result')
        plt.subplot(3,3,7)
        plt.imshow(Sawgrass_all_D4_pred, cmap='binary')
        plt.axis('off')
        plt.title('Sawgrass_all_D4_pred Result')
        plt.show()
