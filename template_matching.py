import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import tqdm

data_dir = '/media/tungthanhlee/DATA/tienthanh/assigned jpeg'
out_dir = '/media/tungthanhlee/DATA/tienthanh/cropped jpeg'
os.makedirs(out_dir, exist_ok=True)
img_names = next(os.walk(data_dir))[2]
# All the 6 methods for comparison in a list
templates = []
templates.append(cv.imread('template4.jpg',0))
templates.append(cv.imread('template6.jpg',0))
templates.append(cv.imread('template8.jpg',0))
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
for i, img_name in enumerate(tqdm.tqdm(img_names)):
    # print(img_name)
    img = cv.imread(os.path.join(data_dir, img_name),0)
    # ow, oh = img.shape[::-1]
    method = eval(methods[5])
    found = None
    ow, oh = img.shape[::-1]
    if ow > 3300 or oh > 3300:
        template = templates[0]
    elif ow < 1100 or oh < 1100:
        template = templates[2]
    else:
        template = templates[1]
    w, h = template.shape[::-1]
    if ow<w or oh<h:
        r = max(w/ow, h/oh)
        nh = math.ceil(r*oh)
        nw = math.ceil(r*ow)
        img = cv.resize(img, (nw, nh))
    # img2 = img.copy()s
    # img = img2.copy()
    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # if found is None or max_val > found[0]:
    #     found = (max_val, max_loc, img, w, h)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    img = img[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
    cv.imwrite(os.path.join(out_dir, img_name), img)
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # print(top_left, bottom_right)
    # print(img.shape)

    # cv.rectangle(img,top_left, bottom_right, 255, 2)
    # plt.figure(i)
    # plt.subplot(121),plt.imshow(res,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img,cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.suptitle(methods[3])
    # plt.show()