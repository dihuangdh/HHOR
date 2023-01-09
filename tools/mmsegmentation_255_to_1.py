'''
Date: 2022-12-21 4:09:18 pm
Author: dihuangdh
Descriptions: 
-----
LastEditTime: 2022-12-21 5:17:20 pm
LastEditors: dihuangdh
'''



if __name__ == '__main__':
    import os
    import cv2
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/val/label')
    parser.add_argument('--output_dir', type=str, default='data/val/label')
    args = parser.parse_args()

    for file in os.listdir(args.input_dir):
        img = cv2.imread(os.path.join(args.input_dir, file), 0)
        img = np.where(img == 255, 1, img)
        cv2.imwrite(os.path.join(args.output_dir, file), img)