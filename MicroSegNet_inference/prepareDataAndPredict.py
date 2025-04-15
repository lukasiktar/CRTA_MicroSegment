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
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from natsort import natsorted
from scipy.interpolate import splprep, splev


def prepare_data_and_predict(main_dir, images_dir, net):
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
    predictions=[]

    output_dir = f"{images_dir}_processed/masks" # Directory where output images will be saved
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    output_dir1 = f"{images_dir}_processed/segmentations" # Directory where segmentation images will be saved
    os.makedirs(output_dir1, exist_ok=True) 

    # output_dir2 = f"{images_dir}_processed/input_images"  # Directory where input images will be saved
    # os.makedirs(output_dir2, exist_ok=True) 

    for counter, image_path in enumerate(natsorted(images_paths)):

        #orig_image=cv2.imread(image_path)
        dicom_image_data=pydicom.dcmread(image_path)
        image_data=dicom_image_data.pixel_array
        orig_image=cv2.normalize(image_data, None, 0,255, cv2.NORM_MINMAX)
        #print(image_path)

        path=image_path.split("_")[-2] 
        deg=image_path.split("_")[-1].split(".d")[-2]
        #deg=image_path.split("/")[-1].split(".j")[-2].split("'")[-1]
        #deg=image_path.split("-")[-1].split(".p")[0]
        deg=deg.replace(f"decdeg","deg")        #If images have wrong name, change it
        deg=int(deg)
        
        #Convert image to grayscale
        image= orig_image

             # Convert the image to a PyTorch tensor
        image_tensor = torch.from_numpy(image).float()          # Convert to float32 tensor
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)   # Add batch and channel dimensions

        #Resize the tensor
        resized_image=cv2.resize(image, (224,224))

        #Input to the neural network
        input_nn = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).float().cuda()

        with torch.no_grad():
                #Model outputs
                outputs, _, _, _, cls_output= net(input_nn)
                #Model predictions
                # Apply sigmoid to classification output
                cls_pred = torch.sigmoid(cls_output).squeeze()
                #print(cls_pred)
                #Check if the classification predicts an object (you may need to adjust the threshold)
                print(cls_pred.item())
                if cls_pred.item() < 0.9:  # Assuming 0.5 as the threshold
                    print("!!!!!!!!!!!!!!!!!!")
                    # If no object is predicted, create a black mask
                    pred = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                   
                else:
                    # If an object is predicted, proceed with segmentation as before
                    out = torch.sigmoid(outputs).squeeze()
                    pred = out.cpu().detach().numpy()

                # if x != patch_size[0] or y != patch_size[1]:
                #     pred = cv2.resize(out, (y, x), interpolation = cv2.INTER_NEAREST)
                
                #Create a binary mask from predicitons to extract only important part of contur (without padding)
                a = 1.0*(pred>0.5)
                prediction = a.astype(np.uint8)
                non_black_mask = (image > 5).astype(np.uint8)
                non_black_mask = cv2.resize(non_black_mask, (prediction.shape[1], prediction.shape[0]))
                prediction = cv2.bitwise_and(prediction, prediction, mask=non_black_mask)

                prediction = cv2.normalize(prediction, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                predictions.append(prediction)
                
                # Optional pre-blur to reduce jagged edges
                prediction = cv2.GaussianBlur(prediction, (5, 5), sigmaX=2)

                # Threshold to ensure binary mask
                _, binary_prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)

                # Find contours on predicted binary masks (224x224)
                contours, _ = cv2.findContours(binary_prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Scale contours to original image size
                orig_h, orig_w = orig_image.shape[:2]
                scale_x = orig_w / 224
                scale_y = orig_h / 224
                scaled_contours = []
                for contour in contours:
                    contour = contour.astype(np.float32)

                    contour[:, 0, 0] *= scale_x
                    contour[:, 0, 1] *= scale_y

                    scaled_contours.append(contour.astype(np.int32))


                smoothed_contours=[]
                
                for contour in scaled_contours:
                     
                    x,y = contour.T
                    x=x.tolist()[0]
                    y=y.tolist()[0]


                    #Use a higher smoothing factor and more interpolation points
                    tck, u = splprep([x, y], s=10.0, k=1, per=True)
                    u_new = np.linspace(u.min(), u.max(), 100)
                    x_new, y_new = splev(u_new, tck, der=0)

                    smoothed_contour = np.array([[int(px), int(py)] for px, py in zip(x_new, y_new)], dtype=np.int32)
                    smoothed_contour = smoothed_contour.reshape((-1, 1, 2))
                    smoothed_contours.append(smoothed_contour)

                #Draw red contour on original image
                segmented_image = orig_image.copy()
                segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(segmented_image, smoothed_contours, -1, (0, 0, 255), 3)

                #Fill in mask using smoothed contour
                filled_mask = np.zeros_like(prediction)
                cv2.drawContours(filled_mask, smoothed_contours, -1, 255, thickness=cv2.FILLED)



                # ================================================
                # NEW DICOM SAVING LOGIC (REPLACE PNG SAVE)
                # ================================================
                # Load original DICOM metadata
                ds_in = pydicom.dcmread(image_path)

                # Create File Meta Information
                file_meta = pydicom.Dataset()
                file_meta.FileMetaInformationVersion = b'\x00\x01'  # Required empty byte string
                file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.66.4'  # Segmentation Storage
                file_meta.MediaStorageSOPInstanceUID = generate_uid()
                file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID  # Use pydicom's UID

                # Create new dataset with file meta
                ds_seg = pydicom.Dataset()
                ds_seg.file_meta = file_meta

                # Add required DICOM attributes from file meta
                ds_seg.SOPClassUID = file_meta.MediaStorageSOPClassUID
                ds_seg.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

                # Copy critical study/patient data from original
                for tag in ['StudyInstanceUID', 'SeriesInstanceUID', 
                            'PatientID', 'PatientName', 'PatientBirthDate',
                            'PatientSex', 'StudyDate', 'StudyTime']:
                    if tag in ds_in:
                        setattr(ds_seg, tag, getattr(ds_in, tag))

                # Configure image parameters
                ds_seg.Rows = filled_mask.shape[0]
                ds_seg.Columns = filled_mask.shape[1]
                ds_seg.PhotometricInterpretation = 'MONOCHROME2'
                ds_seg.SamplesPerPixel = 1
                ds_seg.BitsAllocated = 8
                ds_seg.BitsStored = 8
                ds_seg.HighBit = 7
                ds_seg.PixelRepresentation = 0
                ds_seg.ContentLabel = "PROSTATE_SEGMENTATION"
                ds_seg.SegmentationType = "BINARY"

                # Convert mask to DICOM-compatible format
                seg_array = np.ascontiguousarray(filled_mask, dtype=np.uint8)
                ds_seg.PixelData = seg_array.tobytes()

                # Set required VR encoding
                ds_seg.is_little_endian = True
                ds_seg.is_implicit_VR = False  # Must be explicit VR for segmentation

                # Validate and save
                pydicom.dataset.validate_file_meta(ds_seg.file_meta, enforce_standard=True)
                output_path_dcm = os.path.join(output_dir, f"mask_{dir_name}_{deg}.dcm")
                ds_seg.save_as(output_path_dcm, write_like_original=False)


                output_path1 = os.path.join(output_dir1, f"segmentation_{dir_name}_{deg}.png")
                cv2.imwrite(output_path1, segmented_image)

                # #Smooth the contours
                # smoothened = []
                # for contour in contours:
                    
                #     x_1,y_1 = contour.T
                #     # Convert from numpy arrays to normal arrays
                #     x_1 = x_1.tolist()[0]
                #     y_1 = y_1.tolist()[0]
                #     # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
                #     tck, u = splprep([x_1,y_1], u=None, s=0.0, k=1, per=1)
                #     # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
                #     u_new = np.linspace(u.min(), u.max(), 50)

                    
                #     # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
                #     x_new, y_new = splev(u_new, tck, der=0)
                #     # Convert it back to numpy format for opencv to be able to display it
                #     res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
                #     smoothened.append(np.asarray(res_array, dtype=np.int32))

                # #Show only the biggest contour
                # cv2.drawContours(orig_image, smoothened, 0, (255,255,255),3)

                # #Smooth the conoturs and store them
                # prediction[:] = 0  
                # cv2.drawContours(prediction, smoothened, 0, (255,255,255),-1)
                # cv2.imwrite(output_path, prediction)

                # #Store the images with found contours
                # #output_path1 = os.path.join(output_dir1, f"{dir_name}_segmentation_{path}.png")
                # output_path1 = os.path.join(output_dir1, f"segmentation_{deg:.10f}.png")

                # cv2.imwrite(output_path1, orig_image)


                
                

