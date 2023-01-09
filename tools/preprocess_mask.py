'''
Date: 2021-10-20 8:10:54 pm
Author: dihuangdh
Descriptions: 
-----
LastEditTime: 2022-08-10 2:19:44 pm
LastEditors: dihuangdh
'''
import os
import cv2
import numpy as np
from tqdm import tqdm
import json
from easymocap.mytools.camera_utils import read_camera
from easymocap.mytools.utils import Timer


def read_json(jsonname):
    with open(jsonname) as f:
        data = json.load(f)
    return data


def PCA(dataMat, topNfeat=1):
    meanVals = np.mean(dataMat, axis=0)
    dataMean = dataMat - meanVals
    conMat = dataMean.T.dot(dataMean)
    eigVals, eigVects = np.linalg.eig(conMat)
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    redeigVects = eigVects[:, eigValInd]
    lowdataMat = dataMean.dot(redeigVects)
    condata = (lowdataMat.dot(redeigVects.T)) + meanVals
    reducedata = lowdataMat + np.mean(dataMat)
    return reducedata, condata


def crop_hand_part(img, A, B, C, is_up):
    height, width = img.shape[0], img.shape[1]

    x = np.arange(width)
    y = np.arange(height)
    xv, yv = np.meshgrid(x, y)
    w = A * xv + B * yv + C
    w[w > 0] = 255.
    w[w < 0] = 0.
    if is_up:
        return w
    else:
        return 255. - w


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='')
    parser.add_argument('--threshold', default=0.5, help='threshold for bgmatting masks')
    args = parser.parse_args()

    masks_folder = os.path.join(args.data_dir, 'masks')
    subs = os.listdir(masks_folder)
    if len(subs) == 1:
        vert_folder = os.path.join(args.data_dir, 'output', 'vertices')
    else:
        vert_folder = os.path.join(args.data_dir, 'output', 'vertices')

    # read cameras
    print('Reading cameras...')
    intri_path = os.path.join(args.data_dir, 'intri.yml')
    extri_path = os.path.join(args.data_dir, 'extri.yml')
    cams = read_camera(intri_path, extri_path)

    for sub in subs:
        print(f'Processing sub: {sub}')

        # camera 
        cam = cams[sub]
        K = cam['K']  # (3, 3)
        R = cam['R']  # (3, 3)
        T = cam['T']  # (3, 1)
        
        mask_folder = os.path.join(masks_folder, sub)
        # img_folder = os.path.join(imgs_folder, sub)
        # imgnames = sorted(os.listdir(img_folder))
        masknames = sorted(os.listdir(mask_folder))
        for idx, maskname in tqdm(enumerate(masknames)):
            # read image
            maskpath = os.path.join(mask_folder, maskname)
            mask = cv2.imread(maskpath, 0)

            vert_name = maskname.replace('.png', '.json')
            vert_data = read_json(os.path.join(vert_folder, vert_name))
            vert_3d = np.array(vert_data[0]['vertices'])
            
            # project
            Rt = np.concatenate([R, T], 1)  # (3, 4)
            vert_2d = np.concatenate([vert_3d, np.ones((vert_3d.shape[0], 1))], 1)
            vert_2d = Rt @ vert_2d.transpose(1, 0)
            vert_2d = K @ vert_2d
            vert_2d = (vert_2d[:2, :] / (vert_2d[2:, :]+1e-5)).transpose(1, 0)

            # generate vertices masks, use PCA
            edge_idx = [78, 79, 38, 92, 108, 117, 118, 119, 120, 234, 239, 122, 121, 214, 215, 279]
            points = [vert_2d[j] for j in edge_idx]
            points = np.array(points)
            with Timer('PCA'):
                _, u_points = PCA(points)
            (y_end, x_end) = mask.shape
            x1, y1, x2, y2 = u_points[0, 0], u_points[0, 1], u_points[-1, 0], u_points[-1, 1]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            with Timer('crop'):
                vert_mask = crop_hand_part(mask, A, B, C, A * vert_2d[745, 0] + B * vert_2d[745, 1] + C > 0)
            
            # fuse mask
            bgmat_mask = mask
            vert_mask[bgmat_mask <= args.threshold] = 0.
            mask = vert_mask
            cv2.imwrite(maskpath, mask)