import shutil
import os
import random
import torch
from time import time
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data.dataset import Dataset
from PIL import Image
import argparse
import cv2
import numpy as np


def generate_splits(src_path, dst_path, fold_list_path, fold_list, split, n_folds, substitute_file_type=".npy"):
    img_type = "only_panels"
    list_files_only_panels = generate_split(src_path, dst_path, fold_list_path, fold_list, split, n_folds, img_type,
                                            substitute_file_type=substitute_file_type)

    img_type = "whole_directory"
    list_files_no_panels = generate_split(src_path, dst_path, fold_list_path, fold_list, split, n_folds, img_type,
                                          skip_files_list=list_files_only_panels,
                                          substitute_file_type=substitute_file_type)


def generate_split(src_path, dst_path, fold_list_path, fold_list, split, n_folds, img_type, skip_files_list=None,
                   substitute_file_type=".npy"):
    if skip_files_list is None:
        skip_files_list = [[] for _ in range(0, n_folds + 1)]
        background_images_limit = -1
    else:
        background_images_limit = 0
        for files in skip_files_list:    
            background_images_limit += len(files)
        #background_images_limit = int(background_images_limit)
        background_images_limit = -1
        

    dst_path_img = os.path.join(dst_path, split, img_type)

    if not os.path.exists(dst_path_img):
        os.mkdir(dst_path_img)
    else:
        shutil.rmtree(dst_path_img)
        os.mkdir(dst_path_img)

    fold_img_list = [[] for _ in range(0, n_folds + 1)]

    print(f'Preparing {img_type} folds: Received {fold_list_path} to be processed from {fold_list} lists')
    i = 0
    for fold_number in fold_list:
        print(f'Processing {fold_list_path}/{img_type}/fold{fold_number}.txt')
        img_file = open(f'{fold_list_path}/{img_type}/fold{fold_number}.txt', "r")
        while True:
            line = img_file.readline()

            if not line:
                break
            else:
                i += 1
                if substitute_file_type:
                    fold_img_list[fold_number].append(line.strip().replace(substitute_file_type, ".png"))
                else:
                    fold_img_list[fold_number].append(line.strip())

        img_file.close()

    print(f'Total imgs ready to add: {i}')

    print(background_images_limit)

    src_path_img = os.path.join(src_path, "img")
    # src_path_mask = os.path.join(src_path, "mask")
    images_processed = 0
    for fold_number in fold_list:
        for img_file in fold_img_list[fold_number]:
            if img_file not in skip_files_list[fold_number]:
                # Gather images and copy to destination path
                src_img = os.path.join(src_path_img, img_file)
                dst_img = os.path.join(dst_path_img, img_file)
                shutil.copy(src_img, dst_img)
                images_processed += 1

            if background_images_limit == images_processed:
                print("here")

            if skip_files_list is not None and background_images_limit == images_processed:
                break 
        
        if skip_files_list is not None and background_images_limit == images_processed:
                break 

    print(f'Total files created: Images - {len(os.listdir(dst_path_img))}')
    print(f'Total images available: Images - {len(os.listdir(src_path_img))}')

    return fold_img_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess dataset for training.')

    parser.add_argument('-sp', '--src_path', default='/home/ad084/ad084_media/ign', type=str,
                        help='source dataset path with mask and img subdirectories where the respective files are stored')
    parser.add_argument('-dp', '--dst_path', default='/home/ad084/ad084_media/ign', type=str,
                        help='destination path for the split')
    parser.add_argument('-flp', '--fold_list_path',
                        default='/home/ad084/solarpv_project/SolarPV/misc/ign/', type=str,
                        help='path to the file folds')
    parser.add_argument('-fl', '--fold_list', default="2,3,4",type=str,
                        help='fold list')
    parser.add_argument('-s', '--split', default="train",type=str,
                        help='fold type to be generated')

    args = parser.parse_args()

    fold_list = [int(fold) for fold in str(args.fold_list).split(",")]

    generate_splits(args.src_path, args.dst_path, args.fold_list_path, fold_list, args.split, 5)
