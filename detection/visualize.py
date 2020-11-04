# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:15:50 2020

@author: Andika Rachman
"""

import os
import cv2
import datetime
import numpy as np
import skimage.draw

# List of clothes classes
category_id = {1: "short sleeve top",
               2: "long sleeve top",
               3: "short sleeve outwear",
               4: "long sleeve outwear",
               5: "vest",
               6: "sling",
               7: "shorts",
               8: "trousers",
               9: "skirt",
               10:"short sleeve dress",
               11:"long sleeve dress",
               12:"vest dress",
               13:"sling dress"}

def color_splash(image, mask):
    """
    Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def draw_bbox(image, rois, class_ids):
    """
    Draw bounding box(es) and class name(s).
    image: RGB image [height, width, 3]
    rois: list of roi/bounding boxes [[ymin1, xmin1, ymax1, xmaax1], [...], ....]
    class_ids: list of predicted class ids [integer, integer, ...]

    Returns result image.
    """
    
    for i, roi in enumerate(rois):
        #text = category_id[class_ids[i]]
        #text_size = cv2.getTextSize(text,cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
        start_point = (roi[1], roi[0])
        end_point = (roi[3], roi[2])

        image = cv2.rectangle(image, start_point, end_point, (51, 51, 255), 8)
        #image = cv2.rectangle(image, start_point, (start_point[0] + text_size[0][0], 
        #                                           start_point[1] + text_size[0][1] + 3), 
        #                      (51, 51, 255), -1)
        #image = cv2.putText(image, text, (start_point[0], start_point[1] + text_size[0][1]), 
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
        #                    1, cv2.LINE_AA)
    return image


def visualize_detection(dt_results, image, det_filename, visualize):
    # Color splash
    splash = color_splash(image, dt_results["masks"])
        
    # Draw bounding box(s)
    splash = draw_bbox(splash, dt_results["rois"], dt_results["class_ids"])
    splash = cv2.cvtColor(splash, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/images/detection/' + det_filename, splash)
    
    if visualize:
        # Visualize results
        cv2.imshow("detection results", splash)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


