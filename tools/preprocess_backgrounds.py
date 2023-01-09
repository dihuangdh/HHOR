'''
Date: 2021-11-02 2:50:53 pm
Author: dihuangdh
Descriptions: 
-----
LastEditTime: 2022-08-10 2:12:50 pm
LastEditors: dihuangdh
'''
import os
import shutil
from tqdm import tqdm


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='')
    args = parser.parse_args()

    imgs_folder = os.path.join(args.data_dir, 'images')
    bgs_folder = os.path.join(args.data_dir, 'backgrounds')
    subs = os.listdir(imgs_folder)
    os.makedirs(bgs_folder, exist_ok=True)

    for sub in subs:
        print(f'Processing sub: {sub}')
        os.makedirs(os.path.join(bgs_folder, sub), exist_ok=True)

        img_folder = os.path.join(imgs_folder, sub)
        bg_folder = os.path.join(bgs_folder, sub)
        length = len(os.listdir(img_folder))

        for idx in tqdm(range(length)):
            shutil.copyfile(
                os.path.join(args.data_dir, 'bg.jpg'),
                os.path.join(bg_folder, f'{idx:06d}.jpg')
            )