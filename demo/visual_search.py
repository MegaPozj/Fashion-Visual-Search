# -*- coding: utf-8 -*-
"""
Created on Sat May 29 11:20:07 2020

@author: Andika Rachman
"""

from __future__ import division
import argparse
import os
import cv2
import random
import sys
import pickle
from datetime import datetime
sys.dont_write_bytecode = True

import torch
import tensorflow as tf
from tensorflow import keras
import mmcv
import skimage.draw
import numpy as np
from PIL import Image

from mmcv.runner import load_checkpoint

from mmfashion.core import ClothesRetriever
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
        description='GetGO Fashion Clothes Retriever Demo')
    parser.add_argument(
        '--image_list', required=False,
        default="references/img_list/img_list.npy",
        help="Path to image reference list")
    parser.add_argument(
        '--image_embeddings', required=False,
        default="references/img_embeds/embeddings.npy",
        help="Path to image embeddings")
    parser.add_argument(
        '--detection_weights', required=False,
        metavar="/path/to/weights.h5",
        default="detection/logs/mask_rcnn_deepfashion2_0005.h5",
        help="Path to weights (.h5 file) of the detection model")
    parser.add_argument(
        '--topk', type=int, default=12, help='retrieve topk items')
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
        '--visualize',
        type=bool,
        default=False,
        help='visualize the retrieved images')
    parser.add_argument(
        '--use_cuda', type=bool, default=False, help='use gpu or not')
    args = parser.parse_args()
    return args


def _process_embeds(embed_file):
    embeds = np.load(embed_file)
    return embeds


def _init_models():
    args = parse_args()

    # Build retrieval model and load checkpoint
    cfg = mmcv.Config.fromfile(args.config_retrieval)
    model_rt = build_retriever(cfg.model)
    load_checkpoint(model_rt, args.checkpoint_retrieval)
    print('load retriever checkpoint from {}'.format(args.checkpoint_retrieval))

    # Build landmark detection model and load checkpoint
    cfg_lm = mmcv.Config.fromfile(args.config_landmark)
    model_lm = build_landmark_detector(cfg_lm.model)
    load_checkpoint(model_lm, args.checkpoint_landmark)
    print('load landmark detector checkpoint from: {}'.format(args.checkpoint_landmark))
    
    if args.use_cuda:
        model_rt.cuda()
        model_lm.cuda()
    model_rt.eval()
    model_lm.eval()

    # Build database for retrieval
    gallery_list = np.load(args.image_list)
    gallery_embeds = _process_embeds(args.image_embeddings)
    retriever = ClothesRetriever(gallery_list, [args.topk])
    print('build database for retrieval')

    # Return retrieval, landmark, and detection model, database for retrieval and retriever
    return model_rt, model_lm, gallery_embeds, retriever


def visual_search(img, models, dt_results, det_filename):

    args = parse_args()

    # Placeholder for results
    results = []
    categories = []

    # Parse the input
    model_rt = models['model_rt']
    model_lm = models['model_lm']
    gallery_embeds = models['gallery_embeds'] 
    retriever = models['retriever'] 

    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # If no detected clothing
    if len(dt_results['rois']) == 0:
            
        # Create tensor from query image
        img_roi = Image.fromarray(img)
        img_tensor = get_img_tensor(img_roi, args.use_cuda)
        class_id = 15

        # Generate landmark from query image tensor
        _, pred_lm = model_lm(img_tensor, return_loss=False)
            
        # Retrieval
        query_feat = model_rt(img_tensor, landmark=pred_lm, return_loss=False)
        query_feat = query_feat.data.cpu().numpy()
        result = retriever.show_retrieved_images(img, query_feat, gallery_embeds, class_id, args.visualize)
        results.append(result)
        categories.append("Clothes")
        
        return results, categories

    # Visualize detection results
    visualize.visualize_detection(dt_results, img, det_filename, args.visualize)

    # Iterate clothes retrieval for every detected ROI 
    for i, roi in enumerate(dt_results['rois']):

        # Cropped input image based on the detected ROI
        img_crop = img[roi[0]:roi[2], roi[1]:roi[3], :]
            
        # Visualize cropped image
        class_id = dt_results["class_ids"][i]
        if (class_id <= 6):
            categories.append('Top')
        elif (class_id >= 10):
            categories.append('One-Piece')
        else:
            categories.append('Bottom')

        # Create tensor from query image
        img_roi = Image.fromarray(img_crop)
        img_tensor = get_img_tensor(img_roi, args.use_cuda)

        # Generate landmark from query image tensor
        _, pred_lm = model_lm(img_tensor, return_loss=False)
            
        # Retrieval
        query_feat = model_rt(img_tensor, landmark=pred_lm, return_loss=False)
        query_feat = query_feat.data.cpu().numpy()
        result = retriever.show_retrieved_images(img_crop, query_feat, gallery_embeds, class_id, args.visualize)
        results.append(result)

    return results, categories

