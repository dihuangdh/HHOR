'''
Date: 2021-11-29 6:57:23 pm
Author: dihuangdh
Descriptions: 
-----
LastEditTime: 2022-09-04 8:52:13 pm
LastEditors: dihuangdh
'''
import os
import os.path as osp
from tqdm import tqdm
import cv2
import numpy as np

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='')

    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    png_dir = osp.join(args.data_dir, 'colmap', 'images_png')
    semantic_dir = osp.join(args.data_dir, 'colmap', 'semantics')
    os.makedirs(semantic_dir, exist_ok=True)

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    pngnames = sorted(os.listdir(png_dir))
    for idx, pngname in enumerate(tqdm(pngnames)):
        pngpath = os.path.join(png_dir, pngname)
        png = cv2.imread(pngpath, cv2.IMREAD_UNCHANGED)

        img = png[:, :, :3].copy()
        mask = png[:, :, 3].copy()
        img[mask!=255] = 0  # BRG

        # test a single image
        result = inference_segmentor(model, img)
        
        skinMask = result[0].astype(np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        # skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        # skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

        # skinMask = cv2.erode(skinMask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations = 2)
        # skinMask = cv2.dilate(skinMask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations = 2)

        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skinMask[skinMask > 0] = 255

        objMask = mask.copy()
        objMask[skinMask==255] = 0

        semantics = np.concatenate([skinMask[..., None], objMask[..., None], mask[..., None]], -1)
        cv2.imwrite(os.path.join(semantic_dir, pngname), semantics) # use .png instead of .jpg; jpg will change the value
        # import ipdb; ipdb.set_trace()
