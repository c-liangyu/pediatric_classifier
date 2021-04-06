import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import tqdm

data_dir = '/media/tungthanhlee/DATA/tienthanh/assigned_jpeg'
out_dir = '/media/tungthanhlee/DATA/tienthanh/cropped jpeg'
os.makedirs(out_dir, exist_ok=True)
img_names = next(os.walk(data_dir))[2]
# All the 6 methods for comparison in a list
# templates = []
# templates.append(cv.imread('template4.jpg',0))
# templates.append(cv.imread('template6.jpg',0))
# templates.append(cv.imread('template8.jpg',0))
template = cv.imread('template7.jpg',0)
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
for i, img_name in enumerate(tqdm.tqdm(img_names)):
    # print(img_name)
    tm = template.copy()
    img = cv.imread(os.path.join(data_dir, img_name),0)
    ow, oh = img.shape[::-1]
    # print(ow, oh)
    if ow>= 2500 or oh>=2500:
        sr = 1/2
    elif (ow > 2000 and ow < 2500) or (oh > 2000 and oh < 2500):
        sr = (1/2 + 1/6)
    elif ow < 1100 or oh < 1100:
        sr = 4/5
    else:
        sr = 4/5
    r = min(ow, oh)*sr
    w, h = tm.shape[::-1]
    r = r / max(w, h)
    w = int(w*r)
    h = int(h*r)
    tm = cv.resize(tm, (w, h))
    method = eval(methods[5])
    # img2 = img.copy()s
    # img = img2.copy()
    # Apply template Matching
    res = cv.matchTemplate(img,tm,method)
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