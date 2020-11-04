import os
from datetime import datetime

import numpy as np
import skimage.draw
from scipy.spatial.distance import cosine as cosine

import matplotlib
try:
    matplotlib.use("macOSX")
except:
    pass
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sys import platform as sys_pf


class ClothesRetriever(object):

    def __init__(self,
                 img_list,
                 topks=[5],
                 extract_feature=False):
        self.topks = topks
        self.img_list = img_list
        self.gallery_idx2im = {}

    def show_topk_retrieved_images(self, retrieved_idxes, query_image, visualize):
        retrieved_imgs = [query_image]
        retrieved_imgs_path = []
        for idx in retrieved_idxes:
            retrieved_img = self.img_list[idx]
            retrieved_imgs_path.append({'fullpath': os.path.join('images', retrieved_img[2:]), 'filename': retrieved_img[2:]})
            if visualize:
                img = skimage.io.imread(retrieved_img)
                img_resized = skimage.transform.resize(img, (200, int(img.shape[1] * 200 / img.shape[0])), anti_aliasing=True)
                retrieved_imgs.append(img_resized)
        if visualize:
            fig = plt.figure(figsize=(10, 10))
            grid = ImageGrid(fig, 111, nrows_ncols=(5, 10), axes_pad=0.1)
            for ax, im in zip(grid, retrieved_imgs):
                ax.axis('off')
                ax.imshow(im)
            plt.show()
        
        return retrieved_imgs_path

    def show_retrieved_images(self, query_image, query_feat, gallery_embeds, class_id, visualize):
        output = []
        query_dist = []
        for i, feat in enumerate(gallery_embeds):
            gallery_class = self.img_list[i].split("/")[3]
            if ((class_id == 1) or (class_id == 2) or (class_id == 5) or (class_id == 6)) and\
                ((gallery_class == "Atasan") or 
                (gallery_class == "Hoodies___Sweatshirts") or 
                (gallery_class == "Jaket___Coat") or 
                # (gallery_class == "Kaos_Kaki___Stocking") or 
                (gallery_class == "KnitWear___Cardigan") or 
                (gallery_class == "Playsuits___Jumpsuits") or 
                (gallery_class == "Plus_Size") or 
                (gallery_class == "Swimwear___Beachwear") or 
                # (gallery_class == "Baju_Tidur___Pakaian_Dalam") or 
                (gallery_class == "Baju_Hamil")):
            
                cosine_dist = cosine(
                    feat.reshape(1, -1), query_feat.reshape(1, -1))
                query_dist.append(cosine_dist)
            
            elif ((class_id == 3) or (class_id == 4)) and\
               ((gallery_class == "Blazer") or 
               (gallery_class == "Jaket___Coat") or 
               (gallery_class == "KnitWear___Cardigan")):
                
                cosine_dist = cosine(
                    feat.reshape(1, -1), query_feat.reshape(1, -1))
                query_dist.append(cosine_dist)

            elif ((class_id == 7) or (class_id == 8) or (class_id == 9)) and\
               ((gallery_class == "Celana_Pendek") or 
               (gallery_class == "Pants___Leggings") or 
               (gallery_class == "Jeans") or 
               (gallery_class == "Rok")):

                cosine_dist = cosine(
                    feat.reshape(1, -1), query_feat.reshape(1, -1))
                query_dist.append(cosine_dist)

            elif ((class_id == 10) or (class_id == 11) or (class_id == 12) or (class_id == 13)) and\
               ((gallery_class == "Dress_copy") or 
               (gallery_class == "Playsuits___Jumpsuits") or 
               (gallery_class == "Baju_Hamil")):

                cosine_dist = cosine(
                    feat.reshape(1, -1), query_feat.reshape(1, -1))
                query_dist.append(cosine_dist)
            
            elif (class_id == 15):
                cosine_dist = cosine(
                    feat.reshape(1, -1), query_feat.reshape(1, -1))
                query_dist.append(cosine_dist)
            
            else:
                query_dist.append(np.inf)

        query_dist = np.array(query_dist)
        order = np.argsort(query_dist)

        for topk in self.topks:
            # print('Retrieved Top %d Results' % topk)
            list_retrieved_images = self.show_topk_retrieved_images(order[:topk], query_image, visualize)
            output.append(list_retrieved_images)
        if(len(output) > 0): 
            output = output[0]
        return output
