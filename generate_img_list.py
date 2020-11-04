import os
import numpy as np

img_list = []
root = "../static/images/data/zalora/"
for path, subdirs, files in os.walk(root):
    for name in files:
        if (name[-3:] == "jpg") or (name[-4:] == "jpeg") or (name[-3:] == "png"):
            fullpath = os.path.join(path, name)
            fn = '/'.join([fullpath.split('/')[0]] + fullpath.split('/')[3:])
            img_list.append(fn)
            print(fn)

#img_list = np.array(img_list)
#img_list.sort()
#np.save('img_list/zalora_swimwear_img_list', included_files)
