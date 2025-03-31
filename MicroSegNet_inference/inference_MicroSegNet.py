# ------------------------------------------------------------------------------
# Script Name: inference_MicroSegNet.py
# Description: Main inference script.
# Author: Luka Siktar
# Date Created: 2025-01-16
# Last Modified: 2025-01-16
# Version: 1.0
# Contact: lsiktar@fsb.hr, luka.siktar@gmail.com
# ------------------------------------------------------------------------------
import argparse
import torch
import sys
sys.path.append("/home/crta-hp-408/PRONOBIS/MicroSegNet")
from CRTA_MicroSegment.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from CRTA_MicroSegment.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from prepareDataAndPredict import prepare_data_and_predict
from displayResults import display_results

MODEL_PATH="model/CRTA_MicroSegmentMicroUS224_R50-ViT-B_16_weight4_epo30_bs4_ev02/epoch_29.pth"

#Real prostate (ExactVu)
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/ExactVU_dataset/20231211074944206"
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/ExactVU_dataset/20231211081900898"
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/ExactVU_dataset/20231211100413984"
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/ExactVU_dataset/20231211121343552"
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/ExactVU_dataset/20231212082240438"
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/ExactVU_dataset/20231212091656762"
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/ExactVU_dataset/20231212095259160_1"
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/MicroSegNet/CRTA_MicroSegment/MicroSegNet_inference/Test_images/microsegnet_data_17"
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/ExactVU_dataset/ExactVu_fantom"
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/CRTA_fantom_dataset/prostate_1_full"
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/CRTA_fantom_dataset/prostate_2_full"
#DATASET_DIRECTORY="/home/crta-hp-408/Downloads/5/processed"
#DATASET_DIRECTORY="/home/crta-hp-408/Downloads/8/processed"
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/CRTA_fantom_dataset/sweep_P1_1_30012025"
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/CRTA_fantom_dataset/sweep_P1_2_30012025"
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/CRTA_fantom_dataset/sweep_P1_3_30012025"
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/CRTA_fantom_dataset/sweep_P2_1_30012025"
DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/CRTA_fantom_dataset/sweep_P2_2_30012025"
#DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/CRTA_fantom_dataset/sweep_P2_3_30012025"

MAIN_DIRECTORY_NAME=DATASET_DIRECTORY.split("/")[-1]
print(MAIN_DIRECTORY_NAME)
INPUT_IMAGES_DIRECTORY=f"{MAIN_DIRECTORY_NAME}/input_images"
OUTPUT_MASKS_DIRECTORY=f"{MAIN_DIRECTORY_NAME}/output_images"
OUTPUT_SEGMENTATIONS_DIRECTORY=f"{MAIN_DIRECTORY_NAME}/output_segmentations"

#Define arguments for MicroSegNet model initialization
parser = argparse.ArgumentParser()
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--num_classes', type=int,default=1, help='output channel of network')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')

args, unknown = parser.parse_known_args()


#Define ViT model and load weights
config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
if args.vit_name.find('R50') !=-1:
    config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

net.load_state_dict(torch.load(MODEL_PATH))
net.eval()
#Prepare the and perform the inference dataset
images_dir=DATASET_DIRECTORY
prepare_data_and_predict(MAIN_DIRECTORY_NAME,images_dir, net)

#Optional: Display the inference results
#display_results(number_of_images_to_show=10, output_dir=OUTPUT_SEGMENTATIONS_DIRECTORY)