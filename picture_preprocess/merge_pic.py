# coding: utf-8
# image_merge.py
# 图片垂直合并
# http://www.redicecn.com
# redice@163.com

import os
from PIL import Image

def image_merge(images):
    max_height = 0
    total_width = 0
    # 计算合成后图片的宽度（以最宽的为准）和高度
    for img_path in images:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            width, height = img.size
            if height > max_height:
                max_height = height
            total_width += width

    # 产生一张空白图
    new_img = Image.new('RGB', (total_width, max_height), 255)
    # 合并
    x = y = 0
    for img_path in images:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            width, height = img.size
            new_img.paste(img, (x, y))
            x += width
    return new_img


if __name__ == '__main__':
    img = image_merge(images=[r'test_pic\1.jpg', r'test_pic\2.JPG', r'test_pic\3.jpg'])
    img.show()
