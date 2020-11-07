import os
import numpy as np

root = "../static/images/data/zalora/"
img_list = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if (name[-3:] == "jpg") or (name[-4:] == "jpeg") or (name[-3:] == "png"):
            fullpath = os.path.join(path, name)
            fn = '/'.join([fullpath.split('/')[0]] + fullpath.split('/')[3:])[1:]
            #fn = fullpath
            img_list.append(fn)
img_list = np.array(img_list)
img_list.sort()
np.save('./img_list/img_list.npy', img_list)
