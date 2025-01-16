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

from CRTA_MicroSegment.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from CRTA_MicroSegment.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from prepareDataAndPredict import prepare_data_and_predict
from displayResults import display_results

MODEL_PATH="model/CRTA_MicroSegmentMicroUS224_R50-ViT-B_16_weight4_epo40_bs4/epoch_39.pth"
DATASET_DIRECTORY="/home/crta-hp-408/PRONOBIS/ExactVU_dataset/20231212095259160_1"
INPUT_IMAGES_DIRECTORY="input_images"
OUTPUT_MASKS_DIRECTORY="output_images"
OUTPUT_SEGMENTATIONS_DIRECTORY="output_segmentations"

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

#Prepare the and perform the inference dataset
images_dir=DATASET_DIRECTORY
prepare_data_and_predict(images_dir, net)

#Optional: Display the inference results
display_results(number_of_images_to_show=10, output_dir=OUTPUT_SEGMENTATIONS_DIRECTORY)