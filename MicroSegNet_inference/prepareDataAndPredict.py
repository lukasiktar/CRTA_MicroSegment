# ------------------------------------------------------------------------------
# Script Name: prepareDataAndPredict.py
# Description: The script used for data preparation and inference. The input images,
#           mask images and segmentation images are stored.
# Author: Luka Siktar
# Date Created: 2025-01-16
# Last Modified: 2025-01-16
# Version: 1.0
# Contact: lsiktar@fsb.hr, luka.siktar@gmail.com
# ------------------------------------------------------------------------------
import os
import cv2
import torch
import numpy as np

def prepare_data_and_predict(images_dir, net):
    #Image size
    dir_name=images_dir.split("/")[-1]
    patch_size=[224, 224]
    x, y = patch_size[0], patch_size[1]

    #Read the file names and create complete image paths
    images_paths=[]
    for file in os.listdir(images_dir):
        full_path=os.path.join(images_dir, file)
        images_paths.append(full_path)

    #Create the ouput directories
    #images=[]
    predictions=[]

    output_dir = 'output_images'  # Directory where output images will be saved
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    output_dir1 = 'output_segmentations'  # Directory where segmentation images will be saved
    os.makedirs(output_dir1, exist_ok=True) 

    output_dir2 = 'input_images'  # Directory where input images will be saved
    os.makedirs(output_dir2, exist_ok=True) 

    for counter, image_path in enumerate(images_paths):

        orig_image=cv2.imread(image_path)
        path=image_path.split("_")[-2] 
        deg=image_path.split("_")[-1].split(".j")[-2]

        #images.append(orig_image)
        
        #Convert image to grayscale
        image= cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)

        #Store the input=original images
        output_path_original_image= os.path.join(output_dir2, f"{dir_name}_original_{path}.png")
        cv2.imwrite(output_path_original_image, image)

        # Convert the image to a PyTorch tensor
        image_tensor = torch.from_numpy(image).float()          # Convert to float32 tensor
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)   # Add batch and channel dimensions

        #Resize the tensor
        resized_image=cv2.resize(image, (224,224))

        #Input to the neural network
        input_nn = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float().cuda()

        with torch.no_grad():
                #Model outputs
                outputs, _, _, _  = net(input_nn)
                #Model predictions
                out = torch.sigmoid(outputs).squeeze()
                pred = out.cpu().detach().numpy()

                if x != patch_size[0] or y != patch_size[1]:
                    pred = cv2.resize(out, (y, x), interpolation = cv2.INTER_NEAREST)
                
                #Create a binary mask from predicitons
                a = 1.0*(pred>0.5)
                prediction = a.astype(np.uint8)
                #Resize to original image (fix - autmate)
                prediction=cv2.resize(prediction, (924,962))
                prediction = cv2.normalize(prediction, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                predictions.append(prediction)

                #Store masks
                output_path = os.path.join(output_dir, f"{dir_name}_slice_{path}.png")
                cv2.imwrite(output_path, prediction)

                #Find contours on predicted masks (used for visualization)
                contours, hierarchy = cv2.findContours(prediction,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(orig_image, contours, -1, (0,0,255),5)

                #Store the images with found contours
                output_path1 = os.path.join(output_dir1, f"{dir_name}_segmentation_{path}.png")
                cv2.imwrite(output_path1, orig_image)


                
                

