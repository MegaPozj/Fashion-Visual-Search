# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:24:56 2020

@author: Andika Rachman
"""

from __future__ import division
import argparse
import os
import cv2
import random
import sys
sys.dont_write_bytecode = True

import torch
import mmcv
import skimage.draw
import numpy as np
from PIL import Image

from mmcv.runner import load_checkpoint

from mmfashion.core import ClothesRetriever
from mmfashion.datasets import build_dataloader, build_dataset
from mmfashion.models import build_retriever, build_landmark_detector
from mmfashion.utils import get_img_tensor

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from detection.lib.config import Config
from detection.lib.model import MaskRCNN
from detection.lib import utils
from detection import visualize


class DetectionConfig(Config):
    """
    Configuration for performing detection.
    Derives from the base Config class.
    """
    NAME = "Fashion Item Detection"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 13 
    USE_MINI_MASK = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Embeddings for Retrieval')
    parser.add_argument(
        '--img_list',
        help='npy file containing list of images to be processed')
    parser.add_argument(
        '--save_dir',
        help='directory to save the embeddings npy files',
        default='img_embeds')
    parser.add_argument(
        '--output_fn',
        help='name of the output npy files')
    parser.add_argument(
        '--detection_weights', required=False,
        metavar="/path/to/weights.h5",
        default="detection/logs/mask_rcnn_deepfashion2_0005.h5",
        help="Path to weights (.h5 file) of the detection model")
    parser.add_argument(
        '--config_retrieval',
        help='clothes retriever config file path',
        default='configs/retriever_consumer_to_shop/roi_retriever_vgg.py')
    parser.add_argument(
        '--config_landmark',
        help='landmark detection config file path',
        default='configs/landmark_detect/landmark_detect_vgg.py')
    parser.add_argument(
        '--checkpoint_retrieval',
        type=str,
        default='checkpoint/customer_retrieve/epoch_100.pth',
        help='the checkpoint file for clothes retrieval')
    parser.add_argument(
        '--checkpoint_landmark',
        type=str,
        default='checkpoint/landmark/vgg/latest.pth',
        help='the checkpoint file for landmark detection')
    parser.add_argument(
        '--use_cuda', type=bool, default=False, help='use gpu or not')
    args = parser.parse_args()
    return args


def main():

    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    args = parse_args()

    # Build retrieval model and load checkpoint
    cfg = mmcv.Config.fromfile(args.config_retrieval)
    model = build_retriever(cfg.model)
    load_checkpoint(model, args.checkpoint_retrieval)
    print('load retriever checkpoint from {}'.format(args.checkpoint_retrieval))

    # Build landmark detection model and load checkpoint
    cfg_lm = mmcv.Config.fromfile(args.config_landmark)
    model_lm = build_landmark_detector(cfg_lm.model)
    load_checkpoint(model_lm, args.checkpoint_landmark)
    print('load landmark detector checkpoint from: {}'.format(args.checkpoint_landmark))
    
    if args.use_cuda:
        model.cuda()
        model_lm.cuda()
    model.eval()
    model_lm.eval()

    # Build model for detection
    config = DetectionConfig()
    model_dt = MaskRCNN(mode="inference", config=config, model_dir="None")
    model_dt.load_weights(args.detection_weights, by_name=True)

    # Import image list to be processed
    img_list = np.load(args.img_list)

    # Initiate empty list to store generated embeddings
    embeds = []

    # filename of the output
    filename = os.path.join(args.save_dir, args.output_fn)

    iteration = 0
    for idx, img_path in enumerate(img_list):
        #if idx <= 3900:
        #    continue

        # Import image
        img = skimage.io.imread(img_path)

        # Extract class name
        class_name = img_path.split("/")[3]
        selected_class = None

        if (len(img.shape) == 2) or (img.shape[2] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Perform detection
        dt_results = model_dt.detect([img], verbose=1)[0]

        print(dt_results["class_ids"])

        # If no detected clothing
        if len(dt_results['rois']) == 0:

            # Create tensor from query image
            img_roi = Image.fromarray(img)
            img_tensor = get_img_tensor(img_roi, args.use_cuda)

            # Generate landmark from image tensor
            _, pred_lm = model_lm(img_tensor, return_loss=False)

            # Generate embedding
            embed = model(img_tensor, landmark=pred_lm, return_loss=False)

            selected_roi = np.array([0, 0, 0, 0])

        elif ((class_name == "Atasan") or 
              (class_name == "Baju_Hamil") or 
              (class_name == "Baju_Tidur___Pakaian_Dalam") or 
              (class_name == "Blazer") or 
              (class_name == "Dress_copy") or 
              (class_name == "Hoodies___Sweatshirts") or 
              (class_name == "Jaket___Coat") or 
              (class_name == "Kaos_Kaki___Stocking") or 
              (class_name == "KnitWear___Cardigan") or 
              (class_name == "Playsuits___Jumpsuits") or 
              (class_name == "Plus_Size") or 
              (class_name == "Swimwear___Beachwear")) and\
             ((1 not in dt_results["class_ids"]) and 
              (2 not in dt_results["class_ids"]) and
              (3 not in dt_results["class_ids"]) and
              (4 not in dt_results["class_ids"]) and
              (5 not in dt_results["class_ids"]) and
              (6 not in dt_results["class_ids"]) and
              (10 not in dt_results["class_ids"]) and
              (11 not in dt_results["class_ids"]) and
              (12 not in dt_results["class_ids"]) and
              (13 not in dt_results["class_ids"])):

              # Create tensor from query image
              img_roi = Image.fromarray(img)
              img_tensor = get_img_tensor(img_roi, args.use_cuda)

              # Generate landmark from image tensor
              _, pred_lm = model_lm(img_tensor, return_loss=False)

              # Generate embedding
              embed = model(img_tensor, landmark=pred_lm, return_loss=False)

              selected_roi = np.array([0, 0, 0, 0])

        elif ((class_name == "Celana_Pendek") or 
              (class_name == "Pants___Leggings") or 
              (class_name == "Jeans") or 
              (class_name == "Rok")) and\
             ((7 not in dt_results["class_ids"]) and 
              (8 not in dt_results["class_ids"]) and
              (9 not in dt_results["class_ids"])):

              # Create tensor from query image
              img_roi = Image.fromarray(img)
              img_tensor = get_img_tensor(img_roi, args.use_cuda)

              # Generate landmark from image tensor
              _, pred_lm = model_lm(img_tensor, return_loss=False)

              # Generate embedding
              embed = model(img_tensor, landmark=pred_lm, return_loss=False)

              selected_roi = np.array([0, 0, 0, 0])

        else:
            max_area = 0
            selected_roi = 0
            selected_class = 0
            
            # Iterate clothes retrieval for every detected ROI 
            for i, roi in enumerate(dt_results['rois']):
                class_id = int(dt_results["class_ids"][i])
                if ((class_name == "Atasan") or 
                    (class_name == "Baju_Hamil") or 
                    (class_name == "Baju_Tidur___Pakaian_Dalam") or 
                    (class_name == "Blazer") or 
                    (class_name == "Dress_copy") or 
                    (class_name == "Hoodies___Sweatshirts") or 
                    (class_name == "Jaket___Coat") or 
                    (class_name == "Kaos_Kaki___Stocking") or 
                    (class_name == "KnitWear___Cardigan") or 
                    (class_name == "Playsuits___Jumpsuits") or 
                    (class_name == "Plus_Size") or 
                    (class_name == "Swimwear___Beachwear")) and\
                    (class_id != 7 and class_id != 8 and class_id != 9):

                    area = (roi[2] - roi[0]) * (roi[3] - roi[1])

                    if area < max_area:
                        continue
                    max_area = area
                    selected_roi = roi
                    selected_class = class_id

                    # Cropped input image based on the detected ROI
                    img_crop = img[roi[0]:roi[2], roi[1]:roi[3], :]

                    # Create tensor from query image
                    img_roi = Image.fromarray(img_crop)
                    img_tensor = get_img_tensor(img_roi, args.use_cuda)

                    # Generate landmark from query image tensor
                    _, pred_lm = model_lm(img_tensor, return_loss=False)

                     # Generate embedding
                    embed = model(img_tensor, landmark=pred_lm, return_loss=False)
                    
                elif ((class_name == "Celana_Pendek") or 
                      (class_name == "Pants___Leggings") or 
                      (class_name == "Jeans") or 
                      (class_name == "Rok")) and\
                      (class_id == 7 or class_id == 8 or class_id == 9):
                      
                      area = (roi[2] - roi[0]) * (roi[3] - roi[1])
                      if area < max_area:
                          continue
                      max_area = area
                      selected_roi = roi
                      selected_class = class_id
                        
                      # Cropped input image based on the detected ROI
                      img_crop = img[roi[0]:roi[2], roi[1]:roi[3], :]

                      # Create tensor from query image
                      img_roi = Image.fromarray(img_crop)
                      img_tensor = get_img_tensor(img_roi, args.use_cuda)

                      # Generate landmark from query image tensor
                      _, pred_lm = model_lm(img_tensor, return_loss=False)

                      # Generate embedding
                      embed = model(img_tensor, landmark=pred_lm, return_loss=False)
                        
        start_point = (selected_roi[1], selected_roi[0])
        end_point = (selected_roi[3], selected_roi[2])
        img = cv2.rectangle(img, start_point, end_point, (51, 51, 255), 2)
        #img = cv2.putText(img, selected_class, (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (400, int(img.shape[0]* 400/img.shape[1])), interpolation = cv2.INTER_AREA)
        print(class_name, selected_class)
        cv2.imshow("detection results", img)
        cv2.waitKey(1000)
        
        embeds.append(embed.detach().numpy())
        print(str(idx) + ". Finish generating embedding for " + img_path)

        if (idx != 0) and (idx % 100 == 0):
            iteration += 1
            np.save(filename + '_' + str(iteration), embeds)
            print('Embeddings {} are saved'.format(iteration))
            embeds = []
        
    iteration += 1
    np.save(filename + '_' + str(iteration), embeds)
    print('Embeddings {} are saved'.format(iteration))
    embeds = []

    embed_combine = []
    for i in range(iteration):
        embeds = np.load(filename + '_' + str(i+1) + '.npy')
        for embed in embeds:
            embed_combine.append(embed)
    np.save(filename, embed_combine)
    print("All embeddings have been combined")


if __name__ == '__main__':
    main()
