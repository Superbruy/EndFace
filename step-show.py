import cv2 as cv
import numpy as np
from PIL import Image

import os

if __name__ == '__main__':
    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "duan", "bad", "Image_20201028135512424.jpg")
    image = Image.open(file_path)

    image = np.asarray(image)
    img_shape = image.shape
    # convert bgr format to gray
    if len(img_shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # gradX = cv.Sobel(image, ddepth=cv.CV_32F, dx=1, dy=0)
    # gradY = cv.Sobel(image, ddepth=cv.CV_32F, dx=0, dy=1)
    # gradient = cv.add(gradX, gradY)
    # gradient = cv.convertScaleAbs(gradient)

    blurred = cv.GaussianBlur(image, (9, 9), 0)
    (_, thresh) = cv.threshold(blurred, 90, 255, cv.THRESH_BINARY)

    # 形态学
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    # 腐蚀膨胀
    closed = cv.erode(closed, None, iterations=1)
    closed = cv.dilate(closed, None, iterations=4)

    # 检测轮廓
    (_, cnts, _) = cv.findContours(
        # 参数一： 二值化图像
        closed.copy(),
        # 参数二：轮廓类型
        cv.RETR_EXTERNAL,  # 表示只检测外轮廓
        # cv2.RETR_CCOMP,                #建立两个等级的轮廓,上一层是边界
        # cv2.RETR_LIST,                 #检测的轮廓不建立等级关系
        # cv2.RETR_TREE,                 #建立一个等级树结构的轮廓
        # cv2.CHAIN_APPROX_NONE,         #存储所有的轮廓点，相邻的两个点的像素位置差不超过1
        # 参数三：处理近似方法
        cv.CHAIN_APPROX_SIMPLE,  # 例如一个矩形轮廓只需4个点来保存轮廓信息
        # cv2.CHAIN_APPROX_TC89_L1,
        # cv2.CHAIN_APPROX_TC89_KCOS
    )

    center = []
    radius = []
    for i in cnts:
        x, y, w, h = cv.boundingRect(i)
        print(x, y, w, h)
        if 1030 <= h <= 1110 and 1030 <= w <= 1110:
            print(w, h, x, y)
            c = (int(x + w / 2), int(y + h / 2))
            r = int(w // 2)
            center.append(c)
            radius.append(r)
    print(center, radius)
    pad = np.zeros(img_shape, dtype=np.uint8)
    cv.circle(pad, center[0], radius[0], 255, -1)



    cv.namedWindow('win', 0)
    cv.imshow('win', pad)
    cv.waitKey(0)
    cv.destroyAllWindows()